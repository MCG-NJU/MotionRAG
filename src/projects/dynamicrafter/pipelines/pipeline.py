from typing import Callable, Dict, List, Optional, Union, Literal

import torch
from einops import repeat, rearrange

from ..DynamiCrafter.lvdm.models.ddpm3d import LatentVisualDiffusion
from ..DynamiCrafter.scripts.evaluation.inference import image_guided_synthesis


class DynamiCrafterPipeline:
    def __init__(
            self,
            model: LatentVisualDiffusion
    ):
        self.model = model

    @torch.no_grad()
    def __call__(
            self,
            image: torch.FloatTensor,
            positive_prompt: str,
            negative_prompt: str,
            dtype: Optional[torch.dtype] = torch.float16,
            height: int = 512,
            width: int = 512,
            num_frames: Optional[int] = 16,
            num_inference_steps: int = 50,
            eta: float = 1.0,
            unconditional_guidance_scale: float = 7.5,
            cfg_img: Optional[float] = None,
            frame_stride: int = 20,
            multiple_cond_cfg: bool = False,
            timestep_spacing: str = "uniform",
            guidance_rescale: float = 0.0,
            *args,
            **kwargs,
    ):
        # image: [B C H W] -> [B C 16 H W]
        b = image.shape[0]
        image = repeat(image, 'b c h w -> b c t h w', t=num_frames)
        shape = [b, self.model.model.diffusion_model.out_channels, num_frames, height // 8, width // 8]
        frames = image_guided_synthesis(
            model=self.model,
            prompts=positive_prompt,
            videos=image,
            noise_shape=shape,
            n_samples=1,
            ddim_steps=int(num_inference_steps),
            ddim_eta=eta,
            unconditional_guidance_scale=float(unconditional_guidance_scale),
            cfg_img=cfg_img,
            fs=int(frame_stride),
            text_input=True,
            multiple_cond_cfg=multiple_cond_cfg,
            loop=False,
            interp=False,
            timestep_spacing=timestep_spacing,
            guidance_rescale=guidance_rescale,
        )

        return rearrange(frames, 'b 1 c t h w -> b t c h w')


class DynamiCrafterPipelineRef(DynamiCrafterPipeline):
    @torch.no_grad()
    def __call__(
            self,
            image: torch.FloatTensor,
            positive_prompt: str,
            negative_prompt: str,
            dtype: Optional[torch.dtype] = torch.float16,
            height: int = 512,
            width: int = 512,
            num_frames: Optional[int] = 16,
            num_inference_steps: int = 50,
            eta: float = 1.0,
            unconditional_guidance_scale: float = 7.5,
            cfg_img: Optional[float] = None,
            frame_stride: int = 20,
            multiple_cond_cfg: bool = False,
            timestep_spacing: str = "uniform",
            guidance_rescale: float = 0.0,
            ref_videos: torch.Tensor = None,
            ref_fusion_type: Literal['mean', 'concat', 'top1', 'weight'] = None,
            metadata: dict = None,
            *args,
            **kwargs,
    ):
        # image: [B C H W] -> [B C 16 H W]
        b = image.shape[0]
        image = repeat(image, 'b c h w -> b c t h w', t=num_frames)
        shape = [b, self.model.model.diffusion_model.out_channels, num_frames, height // 8, width // 8]
        videos = image_guided_synthesis(
            model=self.model,
            prompts=positive_prompt,
            videos=image,
            noise_shape=shape,
            n_samples=1,
            ddim_steps=int(num_inference_steps),
            ddim_eta=eta,
            unconditional_guidance_scale=float(unconditional_guidance_scale),
            cfg_img=cfg_img,
            fs=int(frame_stride),
            text_input=True,
            multiple_cond_cfg=multiple_cond_cfg,
            loop=False,
            interp=False,
            timestep_spacing=timestep_spacing,
            guidance_rescale=guidance_rescale,
            ref_videos=ref_videos,
            ref_fusion_type=ref_fusion_type,
            metadata=metadata,
        )

        return rearrange(videos, 'b 1 c t h w -> b t c h w')
