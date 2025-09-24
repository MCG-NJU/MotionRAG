import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from einops import rearrange
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn.functional import cross_entropy, mse_loss, smooth_l1_loss

from ..base_module import BaseModule, OptimizerCallable, LRSchedulerCallable


class ActionCLIP(BaseModule):
    def __init__(self,
                 optimizer: OptimizerCallable = None,
                 lr_scheduler: LRSchedulerCallable = None,
                 freeze_modules: Iterable[str] = None,
                 full_trainable_modules: Iterable[str] = None,
                 lora_trainable_modules: Iterable[str] = None,
                 lora_rank: int = 64,

                 action_model=None,
                 text_model=None,
                 resampler=None,
                 *args,
                 **kwargs
                 ):
        super().__init__(optimizer, lr_scheduler, freeze_modules, full_trainable_modules,
                         lora_trainable_modules, lora_rank, *args, **kwargs)

        self.action_model = action_model
        self.text_model = text_model
        self.resampler = resampler
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def configure_model_(self) -> None:
        self.text_model.freeze()
        self.text_model.device = self.device

        self.action_model = self.action_model.eval()
        for p in self.action_model.parameters():
            p.requires_grad_(False)

    def forward(self, video: torch.Tensor, text: list[str]) -> torch.Tensor:
        text_emb, _ = self.text_model(text, return_cls_tokens=True)

        action_emb = self.action_model(video)
        action_emb, _ = self.resampler(action_emb, return_cls_tokens=True)

        # all_text_emb = text_emb
        # all_action_emb = action_emb
        all_text_emb = self.all_gather(text_emb, sync_grads=True).reshape(-1, text_emb.shape[-1])
        all_action_emb = self.all_gather(action_emb, sync_grads=True).reshape(-1, action_emb.shape[-1])

        logist = all_text_emb @ all_action_emb.T * self.logit_scale.exp()
        label = torch.arange(logist.shape[0], device=self.device)
        loss = (cross_entropy(logist, label) + cross_entropy(logist.T, label)) / 2

        return loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        text = [m['raw_prompt'] for m in batch['metadata']]
        loss = self.forward(batch['video'], text)

        self.log("train/main_loss", loss, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        text = [m['raw_prompt'] for m in batch['metadata']]
        loss = self.forward(batch['video'], text)

        self.log("val/main_loss", loss)

        return loss

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        text = [m['raw_prompt'] for m in batch['metadata']]
        loss = self.forward(batch['video'], text)

        self.log("test/main_loss", loss)

        return loss


class ConditionTransformer(BaseModule):
    def __init__(self,
                 optimizer: OptimizerCallable = None,
                 lr_scheduler: LRSchedulerCallable = None,
                 freeze_modules: Iterable[str] = None,
                 full_trainable_modules: Iterable[str] = None,
                 lora_trainable_modules: Iterable[str] = None,
                 lora_rank: int = 64,

                 condition_model=None,
                 condition_proj=None,
                 vision_model=None,
                 vision_proj=None,
                 transformer=None,
                 compile: bool = False,
                 condition_pe=None,
                 vision_pe=None,
                 *args,
                 **kwargs
                 ):
        super().__init__(optimizer, lr_scheduler, freeze_modules, full_trainable_modules,
                         lora_trainable_modules, lora_rank, *args, **kwargs)

        self.condition_model = condition_model
        self.condition_proj = nn.Linear(condition_model.dim, transformer.layers[
            0].multihead_attn.kdim) if condition_proj is None else condition_proj
        self.vision_model = vision_model
        self.vision_proj = vision_proj
        self.transformer = transformer
        self.compile = compile
        self.condition_pe = condition_pe
        self.vision_pe = vision_pe

    def configure_model_(self) -> None:
        self.vision_model.device = self.device
        self.condition_model.device = self.device

        if self.compile:
            self.vision_model = torch.compile(self.vision_model, fullgraph=True)
            self.vision_proj = torch.compile(self.vision_proj, fullgraph=True)
            self.condition_model = torch.compile(self.condition_model, fullgraph=True)
            self.condition_proj = torch.compile(self.condition_proj, fullgraph=True)

    def get_mask(self, num_frames: int, frame_tokens: int) -> torch.Tensor:
        mask = torch.ones(num_frames * frame_tokens, num_frames * frame_tokens, device=self.device, dtype=torch.bool)
        for i in range(num_frames):
            mask[i * frame_tokens:(i + 1) * frame_tokens, :(i + 1) * frame_tokens] = False
        return mask

    def encode_condition(self, condition: torch.Tensor | list[str]) -> torch.Tensor:
        condition_emb = self.condition_model(condition)
        condition_emb = self.condition_proj(condition_emb)
        if self.condition_pe is not None:
            return self.condition_pe(condition_emb)
        else:
            return condition_emb  # b l c

    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = images.shape
        vision_emb = self.vision_model(rearrange(images, 'b t c h w -> (b t) c h w'))
        vision_emb = self.vision_proj(vision_emb)
        return rearrange(vision_emb, '(b t) l c -> b t l c', b=b)

    def get_loss(self, pred_emb, emb):
        gt = emb[:, 1:]
        return mse_loss(pred_emb, gt)

    def forward(self, visions: torch.Tensor, condition: torch.Tensor | list[str],
                return_loss: bool = True) -> torch.Tensor:
        vision_emb = self.encode_vision(visions)
        condition_emb = self.encode_condition(condition)
        b, num_frames, frame_tokens, d = vision_emb.shape

        mask = self.get_mask(num_frames - 1, frame_tokens)

        x = rearrange(vision_emb[:, :-1], 'b t l c -> b (t l) c')
        if self.vision_pe is not None:
            x = self.vision_pe(x)

        pred_image_emb = self.transformer(x, condition_emb, mask)
        pred_image_emb = rearrange(pred_image_emb, 'b (t l) c -> b t l c', l=frame_tokens)

        if return_loss:
            return self.get_loss(pred_image_emb, vision_emb)
        else:
            return torch.concat([vision_emb[:, 0:1], pred_image_emb], dim=1)

    @torch.no_grad()
    def autoregressive(self, images: torch.Tensor, condition: torch.Tensor | list[str],
                       return_loss: bool = False, num_frames: int = None) -> torch.Tensor:
        t = images.shape[1] if num_frames is None else num_frames
        # b, t, c, h, w = images.shape
        vision_emb = self.encode_vision(images)
        condition_emb = self.encode_condition(condition)
        frame_tokens = vision_emb.shape[2]

        x = x_0 = vision_emb[:, 0]

        for i in range(1, t):
            mask = self.get_mask(i, frame_tokens)
            x = torch.concat([x_0, x], dim=1) if i > 1 else x
            if self.vision_pe is not None:
                x = self.vision_pe(x)
            x = self.transformer(x, condition_emb, mask)

        x = rearrange(x, 'b (t l) c -> b t l c', t=t - 1)

        if return_loss:
            return self.get_loss(x, vision_emb)
        else:
            return torch.concat([vision_emb[:, 0:1], x], dim=1)


class SkillTransformer(ConditionTransformer):
    def __init__(self,
                 *args,
                 context_weight: float = 0.,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.context_weight = context_weight

    def encode_condition(self, condition: list[list[str]]) -> torch.Tensor:
        num_steps = len(condition[0])
        assert all(len(c) == num_steps for c in condition), "All prompt must have the same length"
        batch_condition = sum(condition, [])

        condition_emb = super().encode_condition(batch_condition)
        condition_emb = rearrange(condition_emb, '(b t) l c -> b (t l) c', t=num_steps)
        return condition_emb  # b l c

    def get_uncond_emb(self):
        image_emb = self.vision_model.get_uncond_emb()
        image_emb = self.vision_proj(image_emb)
        return rearrange(image_emb, '(b t) l c -> b t l c', b=1)

    def get_loss(self, pred_emb, emb):
        gt = emb[:, 1:]
        context_emb = emb[:, 0:1].repeat(1, pred_emb.size(1), 1, 1)
        return mse_loss(pred_emb, gt) + mse_loss(pred_emb, context_emb) * self.context_weight

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        num_steps = random.randint(2, batch['max_steps'])
        loss = self.forward(batch['images'][:, :num_steps], batch['prompts'])

        self.log("train/main_loss", loss, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self.autoregressive(batch['images'], batch['prompts'], return_loss=True)

        self.log("val/main_loss", loss)

        return loss

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.autoregressive(batch['images'], batch['prompts'])


@dataclass
class Loss:
    main: torch.Tensor
    mse: torch.Tensor
    smooth: torch.Tensor


class ActionTransformer(ConditionTransformer):
    def __init__(self, ckpt_path: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sos_token = nn.Parameter(torch.randn(1, self.vision_proj.num_queries,
                                                  self.vision_proj.output_dim) / self.vision_proj.output_dim ** 0.5)

        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path, 'cpu')['state_dict'], strict=False)

    def encode_vision(self, videos: torch.Tensor) -> torch.Tensor:
        b, k, t, c, h, w = videos.shape
        vision_emb = self.vision_model(rearrange(videos, 'b k t c h w -> (b k) t c h w'))
        vision_emb = self.vision_proj(vision_emb)
        return rearrange(vision_emb, '(b k) l c -> b k l c', b=b)

    def encode_condition(self, condition: torch.Tensor) -> torch.Tensor:
        b, k, c, h, w = condition.shape

        condition = rearrange(condition, 'b k c h w -> (b k) c h w')
        condition_emb = super().encode_condition(condition)
        condition_emb = rearrange(condition_emb, '(b k) l c -> b (k l) c', k=k)
        return condition_emb  # b l c

    def get_loss(self, pred_emb: torch.Tensor, emb: torch.Tensor) -> Loss:
        """
        Calculate the loss for the action transformer.
        :param pred_emb: predicted embeddings in the shape of (b, t, l, c)
        :param emb: ground truth embeddings in the shape of (b, t, l, c)
        :return: Loss object
        """
        pred_emb = rearrange(pred_emb, 'b t l c -> (b t) l c')
        emb = rearrange(emb, 'b t l c -> (b t) l c')

        mse = mse_loss(pred_emb, emb)
        smooth = smooth_l1_loss(pred_emb, emb)
        return Loss(main=mse, mse=mse, smooth=smooth)

    def forward(self, visions: torch.Tensor, condition: torch.Tensor | list[str],
                return_loss: bool = True, ignore_ref_loss: bool = False) -> torch.Tensor | Loss:
        vision_emb = self.encode_vision(visions)
        condition_emb = self.encode_condition(condition)
        b, num_frames, frame_tokens, d = vision_emb.shape

        x = torch.concat([self.sos_token.repeat(b, 1, 1), rearrange(vision_emb[:, :-1], 'b t l c -> b (t l) c')], dim=1)
        if self.vision_pe is not None:
            x = self.vision_pe(x)
        x += condition_emb

        vision_mask = self.get_mask(num_frames, frame_tokens)

        pred_image_emb = self.transformer(x, vision_mask)
        pred_image_emb = rearrange(pred_image_emb, 'b (t l) c -> b t l c', l=frame_tokens)

        if return_loss:
            if ignore_ref_loss:
                loss = self.get_loss(pred_image_emb[:, -1:], vision_emb[:, -1:])
            else:
                loss = self.get_loss(pred_image_emb, vision_emb)
            return loss
        else:
            return pred_image_emb

    def batch_forward(self, batch, return_loss: bool = True, ignore_ref_loss: bool = False) -> torch.Tensor | Loss:
        ref_videos = batch['ref_videos']  # b k t c h w
        ref_videos = ref_videos.flip(1)  # reverse the similarity
        videos = torch.concat([ref_videos, batch['video'][:, None]], dim=1)
        ref_images = videos[:, :, 0]

        return self.forward(videos, ref_images, return_loss, ignore_ref_loss)

    def predict(self, batch, do_classifier_free_guidance: bool = False) -> torch.Tensor:
        action_emb = self.batch_forward(batch, return_loss=False)[:, -1]
        if do_classifier_free_guidance:
            uncond_action_emb = self.encode_vision(torch.zeros_like(batch['ref_videos'][:, 0:1]))[:, 0]
            action_emb = torch.cat([uncond_action_emb, action_emb], dim=0)

        return action_emb

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self.batch_forward(batch, return_loss=True, ignore_ref_loss=False)
        self.log("train/main_loss", loss.mse, on_step=True, prog_bar=True)
        self.log("train/smooth", loss.smooth, on_step=True, prog_bar=True)

        return loss.main

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self.batch_forward(batch, return_loss=True, ignore_ref_loss=True)
        self.log("val/main_loss", loss.mse)
        self.log("val/smooth", loss.smooth)
        return loss.main

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self.batch_forward(batch, return_loss=True, ignore_ref_loss=True)
        self.log("test/main_loss", loss.mse)
        self.log("test/smooth", loss.smooth)

        return loss.main
