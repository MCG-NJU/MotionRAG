from typing import Iterable

import torch
from einops import rearrange
from lightning.pytorch.utilities.types import STEP_OUTPUT

from .DynamiCrafter.lvdm.models.ddpm3d import LatentVisualDiffusion, LatentActionDiffusion, LatentConditionTransformers
from .DynamiCrafter.lvdm.modules.attention import CrossAttention
from ..base_module import VideoBaseModule, OptimizerCallable, LRSchedulerCallable
from .pipelines.pipeline import DynamiCrafterPipeline, DynamiCrafterPipelineRef
from .DynamiCrafter.main.utils_train import load_checkpoints


class DynamiCrafter(VideoBaseModule, LatentVisualDiffusion):
    def __init__(self,
                 optimizer: OptimizerCallable = None,
                 lr_scheduler: LRSchedulerCallable = None,
                 freeze_modules: Iterable[str] = None,
                 full_trainable_modules: Iterable[str] = None,
                 lora_trainable_modules: Iterable[str] = None,
                 lora_rank: int = 64,
                 eval_pipeline_call_kwargs: dict = None,

                 load_model_kwargs: dict = None,
                 *args,
                 **kwargs
                 ):
        super().__init__(optimizer, lr_scheduler, freeze_modules, full_trainable_modules,
                         lora_trainable_modules, lora_rank, eval_pipeline_call_kwargs, *args, **kwargs)
        self.load_model_kwargs = load_model_kwargs

    def configure_model_(self) -> None:
        LatentVisualDiffusion.configure_model(self)
        load_checkpoints(self, self.load_model_kwargs)

        self.eval_pipeline = DynamiCrafterPipeline(self)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        b, t, c, h, w = batch['video'].shape
        fps = [m['info'].frame_rate for m in batch['metadata']]
        clip_length = [m['clip_length'] for m in batch['metadata']]

        frame_stride = [max(f * l // t, 1.0) for f, l in zip(fps, clip_length)]
        fps_sampled = [f // fs for f, fs in zip(fps, frame_stride)]

        batch_data = {
            'video':        rearrange(batch['video'], 'b t c h w -> b c t h w'),
            'caption':      [m['raw_prompt'] for m in batch['metadata']],
            'fps':          torch.tensor(fps_sampled, device=self.device),
            'frame_stride': torch.tensor(frame_stride, device=self.device),
        }

        return super().training_step(batch_data, batch_idx)


class DynamiCrafterAction(VideoBaseModule, LatentActionDiffusion):
    def __init__(self,
                 optimizer: OptimizerCallable = None,
                 lr_scheduler: LRSchedulerCallable = None,
                 freeze_modules: Iterable[str] = None,
                 full_trainable_modules: Iterable[str] = None,
                 lora_trainable_modules: Iterable[str] = None,
                 lora_rank: int = 64,
                 eval_pipeline_call_kwargs: dict = None,

                 load_model_kwargs: dict = None,
                 *args,
                 **kwargs
                 ):
        super().__init__(optimizer, lr_scheduler, freeze_modules, full_trainable_modules,
                         lora_trainable_modules, lora_rank, eval_pipeline_call_kwargs, *args, **kwargs)
        self.load_model_kwargs = load_model_kwargs

    def configure_model_(self) -> None:
        LatentActionDiffusion.configure_model(self)
        load_checkpoints(self, self.load_model_kwargs)

        if 'action_proj_ckpt' in self.load_model_kwargs:
            action_proj_ckpt = self.load_model_kwargs.pop('action_proj_ckpt')
            action_proj_ckpt = torch.load(action_proj_ckpt, map_location='cpu')['state_dict']

            self.action_proj_model.load_state_dict(
                {k.replace('resampler.', ''): v for k, v in action_proj_ckpt.items() if 'resampler' in k})

        if 'init_action_kv_from_text' in self.load_model_kwargs:
            assert self.load_model_kwargs.pop('init_action_kv_from_text'), "init_action_kv_from_text must be True"
            for m in self.model.diffusion_model.modules():
                if isinstance(m, CrossAttention):
                    if hasattr(m, 'to_k_a') and not m.mix_attention:
                        m.to_k_a.weight.data = m.to_k.weight.data
                        m.to_v_a.weight.data = m.to_v.weight.data

        self.eval_pipeline = DynamiCrafterPipelineRef(self)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        b, t, c, h, w = batch['video'].shape
        fps = [m['info'].frame_rate for m in batch['metadata']]
        clip_length = [m['clip_length'] for m in batch['metadata']]

        frame_stride = [max(f * l // t, 1.0) for f, l in zip(fps, clip_length)]
        fps_sampled = [f // fs for f, fs in zip(fps, frame_stride)]

        batch_data = {
            'video':        rearrange(batch['video'], 'b t c h w -> b c t h w'),
            'ref_videos':   batch['ref_videos'],
            'caption':      [m['raw_prompt'] for m in batch['metadata']],
            'fps':          torch.tensor(fps_sampled, device=self.device),
            'frame_stride': torch.tensor(frame_stride, device=self.device),
        }

        return super().training_step(batch_data, batch_idx)


class DynamiCrafterCT(VideoBaseModule, LatentConditionTransformers):
    def __init__(self,
                 optimizer: OptimizerCallable = None,
                 lr_scheduler: LRSchedulerCallable = None,
                 freeze_modules: Iterable[str] = None,
                 full_trainable_modules: Iterable[str] = None,
                 lora_trainable_modules: Iterable[str] = None,
                 lora_rank: int = 64,
                 eval_pipeline_call_kwargs: dict = None,

                 load_model_kwargs: dict = None,
                 *args,
                 **kwargs
                 ):
        super().__init__(optimizer, lr_scheduler, freeze_modules, full_trainable_modules,
                         lora_trainable_modules, lora_rank, eval_pipeline_call_kwargs, *args, **kwargs)
        self.load_model_kwargs = load_model_kwargs

    def configure_model_(self) -> None:
        LatentVisualDiffusion.configure_model(self)
        load_checkpoints(self, self.load_model_kwargs)

        if 'condition_transformer_ckpt' in self.load_model_kwargs:
            ckpt_path = self.load_model_kwargs.pop('condition_transformer_ckpt')
            ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
            self.condition_transformer.load_state_dict(ckpt, strict=False)

        if 'module_ckpt' in self.load_model_kwargs:
            module_ckpt = self.load_model_kwargs.pop('module_ckpt')
            module_ckpt = torch.load(module_ckpt, map_location='cpu')['state_dict']
            self.load_state_dict(module_ckpt, strict=False)

        self.eval_pipeline = DynamiCrafterPipelineRef(self)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        b, t, c, h, w = batch['video'].shape
        fps = [m['info'].frame_rate for m in batch['metadata']]
        clip_length = [m['clip_length'] for m in batch['metadata']]

        frame_stride = [max(f * l // t, 1.0) for f, l in zip(fps, clip_length)]
        fps_sampled = [f // fs for f, fs in zip(fps, frame_stride)]

        batch_data = {
            'video':        rearrange(batch['video'], 'b t c h w -> b c t h w'),
            'ref_videos':   batch['ref_videos'],
            'caption':      [m['raw_prompt'] for m in batch['metadata']],
            'fps':          torch.tensor(fps_sampled, device=self.device),
            'frame_stride': torch.tensor(frame_stride, device=self.device),
        }

        return super().training_step(batch_data, batch_idx)
