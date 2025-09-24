from typing import Literal

import torch
from diffusers import StableVideoDiffusionPipeline, AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel, \
    EulerDiscreteScheduler
from diffusers.video_processor import VideoProcessor
from einops import rearrange, repeat
from torchvision.transforms.v2 import ToTensor
from torchvision.transforms.v2.functional import pil_to_tensor
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from src.projects.condition.utils import condition_fusion
from src.utils.pipeline import tensor2PIL


class VideoProcessorDtype(VideoProcessor):
    def __init__(self, dtype, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dtype = dtype

    def preprocess(self, *args, **kwargs):
        return super().preprocess(*args, **kwargs).to(self.dtype)


class TupleTensor(tuple):
    """
    A tuple of tensors that can be used as a tensor.
    """

    def to(self, *args, **kwargs):
        return TupleTensor([t.to(*args, **kwargs) for t in self])

    def cuda(self, *args, **kwargs):
        return TupleTensor([t.cuda(*args, **kwargs) for t in self])

    def cpu(self, *args, **kwargs):
        return TupleTensor([t.cpu(*args, **kwargs) for t in self])

    def repeat_interleave(self, *args, **kwargs):
        return TupleTensor([t.repeat_interleave(*args, **kwargs) for t in self])

    def __getitem__(self, item):
        return super().__getitem__(0).__getitem__(item)

    @property
    def dtype(self):
        return super().__getitem__(0).dtype

    @property
    def shape(self):
        return super().__getitem__(0).shape

    def size(self, dim):
        return super().__getitem__(0).size(dim)

    def to_tuple(self):
        return tuple(self)


class SVDActionPipeline(StableVideoDiffusionPipeline):
    model_cpu_offload_seq = "action_embedder->action_proj_model->image_encoder->unet->vae"

    def __init__(
            self,
            vae: AutoencoderKLTemporalDecoder,
            image_encoder: CLIPVisionModelWithProjection,
            unet: UNetSpatioTemporalConditionModel,
            scheduler: EulerDiscreteScheduler,
            feature_extractor: CLIPImageProcessor,
            action_embedder: torch.nn.Module,
            action_proj_model: torch.nn.Module,
            ref_fusion_type: Literal['mean', 'concat', 'top1', 'weight'] = 'mean',
    ):
        super().__init__(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.register_modules(
            action_embedder=action_embedder,
            action_proj_model=action_proj_model,
        )
        self.video_processor = VideoProcessorDtype(dtype=self.vae.dtype, do_resize=True,
                                                   vae_scale_factor=self.vae_scale_factor)
        self.ref_fusion_type = ref_fusion_type

    @property
    def _execution_device(self):
        return self.unet.device

    def __call__(
            self,
            ref_videos: torch.Tensor = None,
            metadata: dict = None,
            *args,
            **kwargs,
    ):
        action_emb = self.action_embedder(rearrange(ref_videos, 'b k f c h w -> (b k) f c h w'))
        action_emb = rearrange(action_emb, '(b k) t c -> b k t c', b=ref_videos.shape[0])

        action_emb = condition_fusion(action_emb, self.ref_fusion_type,
                                      weight=[b['ref_video_distance'] for b in metadata])

        uncond_action_emb = self.action_embedder(torch.zeros_like(ref_videos[:, 0]))
        # action_emb: b t c
        action_emb = self.action_proj_model(torch.cat([uncond_action_emb, action_emb], dim=0))

        self.action_emb = action_emb
        return super().__call__(*args, **kwargs)

    def _encode_image(
            self,
            *args,
            **kwargs,
    ) -> TupleTensor:
        image_embedding = super()._encode_image(*args, **kwargs)
        return TupleTensor([image_embedding, self.action_emb])


class SVDCTPipeline(SVDActionPipeline):
    model_cpu_offload_seq = "condition_transformer->image_encoder->unet->vae"

    def __init__(
            self,
            vae: AutoencoderKLTemporalDecoder,
            image_encoder: CLIPVisionModelWithProjection,
            unet: UNetSpatioTemporalConditionModel,
            scheduler: EulerDiscreteScheduler,
            feature_extractor: CLIPImageProcessor,
            condition_transformer: torch.nn.Module,
    ):
        super(SVDActionPipeline, self).__init__(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.register_modules(
            condition_transformer=condition_transformer,
        )
        self.video_processor = VideoProcessorDtype(dtype=self.vae.dtype, do_resize=True,
                                                   vae_scale_factor=self.vae_scale_factor)

    def __call__(
            self,
            ref_videos: torch.Tensor = None,
            metadata: dict = None,
            *args,
            **kwargs,
    ):
        image = torch.stack([pil_to_tensor(img) for img in kwargs.get('image')]).to(ref_videos.device, ref_videos.dtype)
        image = image / 127.5 - 1.0 # norm to [-1, 1]
        batch_ = {'ref_videos': ref_videos,
                  'video':      repeat(image, 'b c h w->b t c h w', t=ref_videos.size(2))}
        self.action_emb = self.condition_transformer.predict(batch_, do_classifier_free_guidance=True)

        return super(SVDActionPipeline, self).__call__(*args, **kwargs)
