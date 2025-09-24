# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.schedulers import EulerDiscreteScheduler

from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from src.utils.pipeline import _append_dims, tensor2vid, _resize_with_antialiasing




class BasePipeline(DiffusionPipeline):
    model_cpu_offload_seq = None
    _callback_tensor_inputs = ["latents"]

    def clip_encode_image(self, image, do_cfg=False):
        # Normalize the image with for CLIP input
        image = _resize_with_antialiasing(image, (224, 224))
        image = (image + 1.0) / 2.0
        image = self.clip_image_processor(
            images=image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        image_embeds = self.clip_image_encoder(image).image_embeds
        if do_cfg:
            negative_image_embeds = torch.zeros_like(image_embeds)
            image_embeds = torch.cat([negative_image_embeds, image_embeds])
        image_embeds = image_embeds[:, None, :]
        return image_embeds

    def clip_encode_text(self, postive_text, negative_text=None, do_cfg=False):
        positive_text_inputs = self.clip_text_tokenizer(
            postive_text,
            padding="max_length",
            max_length=self.clip_text_tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        positive_text_inputs.input_ids = positive_text_inputs.input_ids.to(self.clip_text_encoder.device)
        attention_mask = None
        if hasattr(self.clip_text_encoder.config, "use_attention_mask") and self.clip_text_encoder.config.use_attention_mask:
            attention_mask = postive_text.attention_mask.to(self.clip_text_encoder.device)
        positive_text_embeds = self.clip_text_encoder(
            input_ids=positive_text_inputs.input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        # for k, v in positive_text_embeds.items():
        #     print(k, v.shape)
        if do_cfg:
            if negative_text is None:
                negative_text = ""
            negative_text_inputs = self.clip_text_tokenizer(
                negative_text,
                padding="max_length",
                max_length=self.clip_text_tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            negative_text_inputs.input_ids = negative_text_inputs.input_ids.to(self.clip_text_encoder.device)
            attention_mask = None
            if hasattr(self.clip_text_encoder.config, "use_attention_mask") and self.clip_text_encoder.config.use_attention_mask:
                attention_mask = negative_text_inputs.attention_mask.to(self.clip_text_encoder.device)
            negative_text_embeds = self.clip_text_encoder(
                input_ids=negative_text_inputs.input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state
            text_embeds = torch.cat([negative_text_embeds, positive_text_embeds])
        else:
            text_embeds = positive_text_embeds
        return text_embeds

    def _vae_encode_image(
            self,
            image: torch.Tensor,
            do_cfg,
    ):
        image_latents = self.vae.encode(image).latent_dist.mode()

        if do_cfg:
            negative_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([negative_image_latents, image_latents])
        return image_latents

    def _vae_decode_latents(
            self,
            latents: torch.Tensor,
            num_frames: int,
            decode_chunk_size: int,
    ):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)
        latents = 1/self.vae.config.scaling_factor * latents
        accepts_num_frames = "num_frames" in set(inspect.signature(self.vae.forward).parameters.keys())
        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in
            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)
        # [batch*frames, channels, height, width] -> [batch, frames, channels, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:])
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames


    def prepare_latents(
        self, *args, **kwargs
    ):
       raise NotImplementedError

    @torch.no_grad()
    def __call__(
        self,
        image: torch.FloatTensor,
        positive_prompt: str,
        negative_prompt: str,
        height: int = 576,
        width: int = 1024,
        dtype: Optional[torch.dtype] = torch.float16,
        num_frames: Optional[int] = 14,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
    ):
        raise NotImplementedError