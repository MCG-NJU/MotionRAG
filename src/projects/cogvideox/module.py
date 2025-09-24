import logging
from typing import Literal

import torch
import torch.nn.functional as F
from diffusers import CogVideoXImageToVideoPipeline, CogVideoXDPMScheduler, CogVideoXDDIMScheduler
from einops import rearrange

from .pipeline import CogVideoXImageToVideoActionPipeline, CogVideoXImageToVideoCTPipeline
from ..base_module import VideoBaseModule
from ..condition import ActionTransformer
from ..condition.attn_processor import APAdapterCogVideoXAttnProcessor2_0
from ..condition.utils import condition_fusion


class CogVideoX5B(VideoBaseModule):
    def __init__(self, *args, gradient_checkpointing: bool = False, ckpt_path: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_checkpointing = gradient_checkpointing
        self.ckpt_path = ckpt_path

    def configure_model_(self) -> None:
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            "THUDM/CogVideoX-5b-I2V",
            torch_dtype=torch.bfloat16
        )

        scheduler_name = self.eval_pipeline_call_kwargs.pop("scheduler", 'ddim')
        if scheduler_name == 'ddim':
            scheduler = CogVideoXDDIMScheduler.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="scheduler")
        elif scheduler_name == 'dpm':
            scheduler = CogVideoXDPMScheduler.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="scheduler")
        else:
            raise ValueError(f'Unknown scheduler: {scheduler_name}')
        pipe.scheduler = scheduler

        pipe = pipe.to(self.device)
        # pipe.enable_sequential_cpu_offload(gpu_id=self.device.index)
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        pipe.transformer.gradient_checkpointing = self.gradient_checkpointing
        pipe.set_progress_bar_config(disable=True)

        self.pipe: CogVideoXImageToVideoPipeline = pipe
        self.scheduler = CogVideoXDPMScheduler.from_pretrained("THUDM/CogVideoX-5b-I2V", subfolder="scheduler")
        self.text_encoder = pipe.text_encoder
        self.vae = pipe.vae
        self.transformer = pipe.transformer
        self.tokenizer = pipe.tokenizer

        if self.ckpt_path is not None:
            logging.info(f'Loading checkpoint from {self.ckpt_path}')
            checkpoint = torch.load(self.ckpt_path, map_location='cpu')
            self.load_state_dict(checkpoint['state_dict'], strict=False)

    def eval_pipeline(self, image, positive_prompt, negative_prompt, dtype, ref_videos, metadata, *args, **kwargs):
        # denormalize
        image = image / 2 + 0.5
        sample_method = kwargs.pop('sample_method', 'first')

        frames = self.pipe(prompt=positive_prompt,
                           image=image,
                           negative_prompt=negative_prompt,
                           output_type='pt',
                           *args,
                           **kwargs)

        video = frames[0]
        if sample_method == 'first':
            video = video[:, :16, ...]  # use only the first 16 frames
        elif sample_method == 'uniform':
            frame_idx = torch.linspace(0, video.shape[1] - 1, 16).round().long()
            video = video[:, frame_idx, ...]
        elif sample_method is None:
            video = video
        else:
            raise ValueError(f'Unknown sample method: {sample_method}')

        video = video * 2 - 1  # normalize
        return video

    def training_step(self, batch, batch_idx):
        video = batch['video']
        batch_size, num_frames, channels, height, width = video.shape

        prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=batch['prompt'],
            do_classifier_free_guidance=False,
            device=self.device,
        )

        # 2. Prepare timesteps
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=self.device,
                                  dtype=torch.int64)

        # 3. Prepare latents
        video = rearrange(video, 'b f c h w-> b c f h w')
        image = rearrange(video[:, :, 0], 'b c h w-> b c 1 h w')
        image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=self.device)
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=image.dtype)
        noisy_images = image + torch.randn_like(image) * image_noise_sigma[:, None, None, None, None]

        latents = self.vae.encode(video).latent_dist.sample()
        latents = rearrange(latents, 'b c f h w -> b f c h w')
        latents *= self.pipe.vae_scaling_factor_image

        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        image_latents = torch.zeros_like(noisy_latents)
        image_latents_ = self.vae.encode(noisy_images).latent_dist.sample()[:, :, 0]
        image_latents_ *= self.pipe.vae_scaling_factor_image
        image_latents[:, 0] = image_latents_

        latent_model_input = torch.cat([noisy_latents, image_latents], dim=2)

        # 4. Create rotary embeds if required
        image_rotary_emb = (
            self.pipe._prepare_rotary_positional_embeddings(height, width, latents.size(1), self.device)
            if self.pipe.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 5. predict noise model_output
        noise_pred = self.pipe.transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            image_rotary_emb=image_rotary_emb,
        ).sample

        latent_pred = self.scheduler.get_velocity(noise_pred, noisy_latents, timesteps)

        alphas_cumprod = self.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        loss = torch.mean((weights * (latent_pred - latents) ** 2).reshape(batch_size, -1), dim=1)
        loss = loss.mean()
        self.log("train/main_loss", loss, on_step=True, prog_bar=True)

        return loss


class CogVideoX5BAction(CogVideoX5B):
    def __init__(self, *args,
                 adapter_modules: list[str],
                 action_proj_model: torch.nn.Module,
                 action_embedder: torch.nn.Module,
                 ref_fusion_type: Literal['mean', 'concat', 'top1', 'weight'] = 'mean',
                 drop_prob: float = 0.0,
                 adapter_path: str = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter_modules = adapter_modules
        self.action_proj_model = action_proj_model
        self.action_embedder = action_embedder
        self.ref_fusion_type = ref_fusion_type
        self.drop_prob = drop_prob
        self.adapter_path = adapter_path

    def set_attention_processors(self, cross_attention_dim: int):
        attn_dict = {}
        hidden_size = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim

        for name, orig_attn in self.transformer.attn_processors.items():
            if name in self.adapter_modules:
                attn_dict[name] = APAdapterCogVideoXAttnProcessor2_0(hidden_size, cross_attention_dim)
            else:
                attn_dict[name] = orig_attn

        self.transformer.set_attn_processor(attn_dict)
        # # set logging level to avoid printing out unnecessary logs
        logging.getLogger("diffusers.models.attention_processor").setLevel(logging.ERROR)

    def configure_model_(self):
        super().configure_model_()
        self.pipe = CogVideoXImageToVideoActionPipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            transformer=self.transformer,
            scheduler=self.scheduler,
            action_embedder=self.action_embedder,
            action_proj_model=self.action_proj_model,
            ref_fusion_type=self.ref_fusion_type,
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.set_attention_processors(self.action_proj_model.cross_attention_dim)

        if self.adapter_path is not None:
            logging.info(f'Loading adapter from {self.adapter_path}')
            checkpoint = torch.load(self.adapter_path, map_location='cpu')
            self.load_state_dict(checkpoint['state_dict'], strict=False)

    def eval_pipeline(self, image, positive_prompt, negative_prompt, dtype, ref_videos, metadata, *args, **kwargs):
        # denormalize
        image = image / 2 + 0.5
        sample_method = kwargs.pop('sample_method', 'first')

        frames = self.pipe(prompt=positive_prompt,
                           image=image,
                           negative_prompt=negative_prompt,
                           output_type='pt',
                           ref_videos=ref_videos,
                           metadata=metadata,
                           *args,
                           **kwargs)

        video = frames[0]
        if sample_method == 'first':
            video = video[:, :16, ...]  # use only the first 16 frames
        elif sample_method == 'uniform':
            frame_idx = torch.linspace(0, video.shape[1] - 1, 16).round().long()
            video = video[:, frame_idx, ...]
        elif sample_method is None:
            video = video
        else:
            raise ValueError(f'Unknown sample method: {sample_method}')

        video = video * 2 - 1  # normalize
        return video

    def training_step(self, batch, batch_idx):
        action_emb = self.pipe.prepare_action_embeddings(batch['ref_videos'], batch['metadata'])
        action_emb = F.dropout1d(action_emb, p=self.drop_prob)

        self.pipe.action_emb = action_emb
        # action_emb -> pipe._prepare_rotary_positional_embeddings -> image_rotary_emb
        # -> CogVideoXTransformer3DModel -> CogVideoXBlock -> APAdapterCogVideoXAttnProcessor2_0 -> ip_hidden_states

        return super().training_step(batch, batch_idx)


class CogVideoX5BActionTransformer(CogVideoX5BAction):
    def __init__(self, *args,
                 adapter_modules: list[str],
                 condition_transformer: ActionTransformer,
                 drop_prob: float = 0.0,
                 adapter_path: str = None,
                 **kwargs):
        super(CogVideoX5BAction, self).__init__(*args, **kwargs)
        self.adapter_modules = adapter_modules
        self.condition_transformer = condition_transformer
        self.drop_prob = drop_prob
        self.adapter_path = adapter_path

    def configure_model_(self):
        super(CogVideoX5BAction, self).configure_model_()
        self.pipe = CogVideoXImageToVideoCTPipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            transformer=self.transformer,
            scheduler=self.scheduler,
            condition_transformer=self.condition_transformer,
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.set_attention_processors(self.condition_transformer.vision_proj.cross_attention_dim)

        if self.adapter_path is not None:
            logging.info(f'Loading adapter from {self.adapter_path}')
            checkpoint = torch.load(self.adapter_path, map_location='cpu')
            self.load_state_dict(checkpoint['state_dict'], strict=False)

    def training_step(self, batch, batch_idx):
        action_emb = self.pipe.prepare_action_embeddings(batch['ref_videos'], batch['metadata'],
                                                         do_classifier_free_guidance=True, image=batch['video'][:, 0])
        action_emb = F.dropout1d(action_emb, p=self.drop_prob)

        self.pipe.action_emb = action_emb
        # action_emb -> pipe._prepare_rotary_positional_embeddings -> image_rotary_emb
        # -> CogVideoXTransformer3DModel -> CogVideoXBlock -> APAdapterCogVideoXAttnProcessor2_0 -> ip_hidden_states

        return super().training_step(batch, batch_idx)
