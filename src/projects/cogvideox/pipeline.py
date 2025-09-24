from typing import Tuple, Union, Literal

import torch
from diffusers import CogVideoXImageToVideoPipeline, AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, \
    CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from einops import rearrange, repeat
from torchvision.transforms.functional import pil_to_tensor
from transformers import T5Tokenizer, T5EncoderModel

from src.projects.condition.utils import condition_fusion


class CogVideoXImageToVideoActionPipeline(CogVideoXImageToVideoPipeline):
    model_cpu_offload_seq = "action_embedder->action_proj_model->text_encoder->transformer->vae"

    def __init__(
            self,
            tokenizer: T5Tokenizer,
            text_encoder: T5EncoderModel,
            vae: AutoencoderKLCogVideoX,
            transformer: CogVideoXTransformer3DModel,
            scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
            action_embedder: torch.nn.Module,
            action_proj_model: torch.nn.Module,
            ref_fusion_type: Literal['mean', 'concat', 'top1', 'weight'] = 'mean',
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.register_modules(
            action_embedder=action_embedder,
            action_proj_model=action_proj_model,
        )
        # self.video_processor = VideoProcessorDtype(dtype=self.vae.dtype, do_resize=True,
        #                                            vae_scale_factor=self.vae_scale_factor)
        self.ref_fusion_type = ref_fusion_type

    @property
    def _execution_device(self):
        return self.transformer.device

    def _prepare_rotary_positional_embeddings(
            self,
            height: int,
            width: int,
            num_frames: int,
            device: torch.device,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Prepare rotary positional embeddings and action embedding
        """
        assert hasattr(self, 'action_emb'), "action_emb is not set"
        return super()._prepare_rotary_positional_embeddings(height, width, num_frames, device), self.action_emb

    def prepare_action_embeddings(self, ref_videos: torch.Tensor, metadata: list[dict],
                                  do_classifier_free_guidance: bool = False, *args, **kwargs) -> torch.Tensor:
        """
        Prepare action embeddings
        :param ref_videos: batch k_ref frame channel height width
        :param metadata: list[dict]
        :param do_classifier_free_guidance: whether to use classifier free guidance
        :return: action_emb: b t c or 2b t c
        """
        action_emb = self.action_embedder(rearrange(ref_videos, 'b k f c h w -> (b k) f c h w'))
        action_emb = rearrange(action_emb, '(b k) t c -> b k t c', b=ref_videos.shape[0])

        action_emb = condition_fusion(action_emb, self.ref_fusion_type,
                                      weight=[b['ref_video_distance'] for b in metadata])

        if do_classifier_free_guidance:
            uncond_action_emb = self.action_embedder(torch.zeros_like(ref_videos[:, 0]))
            action_emb = torch.cat([uncond_action_emb, action_emb], dim=0)
        action_emb = self.action_proj_model(action_emb)
        return action_emb  # b t c

    def __call__(
            self,
            ref_videos: torch.Tensor = None,
            metadata: dict = None,
            *args,
            **kwargs,
    ):
        self.action_emb = self.prepare_action_embeddings(ref_videos, metadata, do_classifier_free_guidance=True, *args,
                                                         **kwargs)
        return super().__call__(*args, **kwargs)


class CogVideoXImageToVideoCTPipeline(CogVideoXImageToVideoActionPipeline):
    model_cpu_offload_seq = "condition_transformer->text_encoder->transformer->vae"

    def __init__(
            self,
            tokenizer: T5Tokenizer,
            text_encoder: T5EncoderModel,
            vae: AutoencoderKLCogVideoX,
            transformer: CogVideoXTransformer3DModel,
            scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
            condition_transformer: torch.nn.Module,
    ):
        super(CogVideoXImageToVideoActionPipeline, self).__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.register_modules(
            condition_transformer=condition_transformer,
        )
        # self.video_processor = VideoProcessorDtype(dtype=self.vae.dtype, do_resize=True,
        #                                            vae_scale_factor=self.vae_scale_factor)

    def prepare_action_embeddings(self, ref_videos: torch.Tensor, metadata: list[dict],
                                  do_classifier_free_guidance: bool = False, *args, **kwargs) -> torch.Tensor:
        """
        Prepare action embeddings
        :param ref_videos: batch k_ref frame channel height width
        :param metadata: list[dict]
        :param do_classifier_free_guidance: whether to use classifier free guidance
        :return: action_emb: b t c or 2b t c
        """
        image = kwargs.get('image').to(ref_videos.device, ref_videos.dtype)
        batch_ = {'ref_videos': ref_videos,
                  'video': repeat(image, 'b c h w->b t c h w', t=ref_videos.size(2))}
        action_emb = self.condition_transformer.predict(batch_, do_classifier_free_guidance=do_classifier_free_guidance)
        return action_emb  # b t c
