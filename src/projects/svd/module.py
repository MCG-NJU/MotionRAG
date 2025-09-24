from typing import Iterable, Callable, Literal

import torch
from diffusers import AutoencoderKLTemporalDecoder, StableVideoDiffusionPipeline
from einops import repeat, rearrange
from torchvision.transforms.v2 import Normalize
import torch.nn.functional as F

from src.projects.base_module import VideoBaseModule, OptimizerCallable, LRSchedulerCallable
from src.projects.condition.attn_processor import APAdapterAttnProcessor2_0
from src.projects.condition.utils import condition_fusion
from src.projects.svd.pipelines.pipeline import SVDActionPipeline, TupleTensor, SVDCTPipeline, VideoProcessorDtype
from src.utils.common import tensor2latent
from src.utils.pipeline import tensor2images, _resize_with_antialiasing, tensor2PIL


class SVDModule(VideoBaseModule):
    def __init__(self,
                 optimizer: OptimizerCallable = None,
                 lr_scheduler: LRSchedulerCallable = None,
                 freeze_modules: Iterable[str] = None,
                 full_trainable_modules: Iterable[str] = None,
                 lora_trainable_modules: Iterable[str] = None,
                 lora_rank: int = 64,
                 eval_pipeline_call_kwargs: dict = None,

                 pretrained_model_path: str = None,
                 condition_noise_config: dict = None,
                 latents_noise_config: dict = None,
                 ):
        super().__init__(optimizer, lr_scheduler, freeze_modules, full_trainable_modules, lora_trainable_modules,
                         lora_rank, eval_pipeline_call_kwargs)
        self.pretrained_model_path = pretrained_model_path
        self.condition_noise_config = condition_noise_config
        self.latents_noise_config = latents_noise_config

    def load_pretrain_model(self):
        pipe = StableVideoDiffusionPipeline.from_pretrained(self.pretrained_model_path)
        self.pipe = pipe
        self.scheduler = pipe.scheduler
        self.vae = pipe.vae
        self.unet = pipe.unet
        self.image_encoder = pipe.image_encoder
        self.feature_extractor = pipe.feature_extractor
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.video_processor = VideoProcessorDtype(dtype=self.vae.dtype, do_resize=True,
                                                        vae_scale_factor=self.pipe.vae_scale_factor)

    def eval_pipeline(
            self,
            image,
            positive_prompt,
            negative_prompt,
            dtype,
            ref_videos,
            metadata,
            *args,
            **kwargs):
        self.pipe.video_processor.dtype = self.pipe.vae.dtype
        self.pipe.scheduler.alphas_cumprod = self.pipe.scheduler.alphas_cumprod.cpu()

        image = tensor2PIL(image)
        frames = self.pipe(
            image=image,
            output_type='pt',
            *args,
            **kwargs,
        ).frames[:, :16]
        return frames * 2 - 1

    def configure_model_(self) -> None:
        self.load_pretrain_model()

    def encode_hidden_states(self, batch):
        # Compute the image embedding
        image = _resize_with_antialiasing(batch["video"][:, 0].to(torch.float32), (224, 224))
        image = (image + 1.0) / 2.0
        image = Normalize(self.feature_extractor.image_mean, self.feature_extractor.image_std)(image)
        image_embeddings = self.image_encoder(image).image_embeds.unsqueeze(1)
        return image_embeddings

    def training_step(self, batch, batch_idx):
        video = batch["video"]
        b, f, c, h, w = video.shape
        # latents: Tensor[b f c h w]
        latents = tensor2latent(video, self.vae)

        image_condition, noise_aug_strength = image2condition_latent(video[:, 0], self.vae,
                                                                     **self.condition_noise_config)
        image_condition = repeat(image_condition, 'b c h w -> b f c h w', f=f)

        sigmas = log_normal((b, 1, 1, 1, 1), **self.latents_noise_config, device=self.device, dtype=self.dtype)
        c_skip = 1 / (sigmas ** 2 + 1)
        c_out = -sigmas / ((sigmas ** 2 + 1) ** 0.5)
        c_in = 1 / ((sigmas ** 2 + 1) ** 0.5)
        c_noise = 0.25 * sigmas.log()

        timesteps = c_noise.squeeze()

        noise = torch.randn_like(latents)
        noise_latents = latents + noise * sigmas

        input_latents = torch.cat([noise_latents * c_in, image_condition], dim=2)

        # Compute the image embedding
        encoder_hidden_states = self.encode_hidden_states(batch)

        # Compute the additional time ids
        added_time_ids = get_add_time_ids(fps=6, motion_bucket_id=127, noise_aug_strength=noise_aug_strength,
                                          device=self.device, dtype=self.dtype)

        pred = self.unet(
            input_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_time_ids=added_time_ids,
        ).sample

        denoised_latents = c_skip * noise_latents + c_out * pred
        weight = (1 + sigmas ** 2) * (sigmas ** -2)

        # Compute main loss
        loss = torch.mean(weight * (denoised_latents - latents) ** 2)

        self.log("train/main_loss", loss, on_step=True, prog_bar=True)

        return loss


class SVDActionModule(SVDModule):
    def __init__(self, *args,
                 adapter_modules: list[str],
                 action_proj_model: torch.nn.Module,
                 action_embedder: torch.nn.Module,
                 ref_fusion_type: Literal['mean', 'concat', 'top1', 'weight'] = 'mean',
                 drop_prob: float = 0.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter_modules = adapter_modules
        self.action_proj_model = action_proj_model
        self.action_embedder = action_embedder
        self.ref_fusion_type = ref_fusion_type
        self.drop_prob = drop_prob

    def set_attention_processors(self):
        attn_dict = {}
        for name, orig_attn in self.unet.attn_processors.items():
            if name in self.adapter_modules:
                cross_attention_dim = self.unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = self.unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = self.unet.config.block_out_channels[block_id]
                else:
                    raise ValueError(f"Attention processor {name} not recognized.")

                attn_dict[name] = APAdapterAttnProcessor2_0(hidden_size, cross_attention_dim)
            else:
                attn_dict[name] = orig_attn

        self.unet.set_attn_processor(attn_dict)
        # # set logging level to avoid printing out unnecessary logs
        # logging.getLogger("diffusers.models.attention_processor").setLevel(logging.ERROR)

    def eval_pipeline(
            self,
            image,
            positive_prompt,
            negative_prompt,
            dtype,
            ref_videos,
            metadata,
            *args,
            **kwargs):
        self.pipe.video_processor.dtype = self.pipe.vae.dtype
        self.pipe.scheduler.alphas_cumprod = self.pipe.scheduler.alphas_cumprod.cpu()

        image = tensor2PIL(image)
        frames = self.pipe(
            image=image,
            ref_videos=ref_videos,
            metadata=metadata,
            output_type='pt',
            *args,
            **kwargs,
        ).frames[:, :16]
        return frames * 2 - 1

    def load_pretrain_model(self):
        super().load_pretrain_model()
        pipe = SVDActionPipeline(vae=self.vae,
                                 image_encoder=self.image_encoder,
                                 unet=self.unet,
                                 scheduler=self.scheduler,
                                 feature_extractor=self.feature_extractor,
                                 action_embedder=self.action_embedder,
                                 action_proj_model=self.action_proj_model,
                                 ref_fusion_type=self.ref_fusion_type)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    def configure_model_(self):
        self.load_pretrain_model()

        self.set_attention_processors()

    def encode_hidden_states(self, batch):
        image_embedding = super().encode_hidden_states(batch)

        ref_videos = batch["ref_videos"]  # batch k_ref frame channel height width

        action_emb = self.action_embedder(rearrange(ref_videos, 'b k f c h w -> (b k) f c h w'))
        action_emb = rearrange(action_emb, '(b k) t c -> b k t c', b=ref_videos.shape[0])

        action_emb = condition_fusion(action_emb, self.ref_fusion_type,
                                      weight=[b['ref_video_distance'] for b in batch["metadata"]])
        # action_emb: b t c
        action_emb = self.action_proj_model(action_emb)
        action_emb = F.dropout1d(action_emb, p=self.drop_prob)
        return TupleTensor([image_embedding, action_emb])


class SVDCTModule(SVDActionModule):
    def __init__(self, *args,
                 adapter_modules: list[str],
                 condition_transformer: torch.nn.Module,
                 load_model_kwargs: dict,
                 **kwargs):
        super(SVDActionModule, self).__init__(*args, **kwargs)
        self.adapter_modules = adapter_modules
        self.condition_transformer = condition_transformer
        self.load_model_kwargs = load_model_kwargs

    def load_pretrain_model(self):
        super(SVDActionModule, self).load_pretrain_model()
        pipe = SVDCTPipeline(vae=self.vae,
                             image_encoder=self.image_encoder,
                             unet=self.unet,
                             scheduler=self.scheduler,
                             feature_extractor=self.feature_extractor,
                             condition_transformer=self.condition_transformer)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    def configure_model_(self):
        self.load_pretrain_model()

        self.set_attention_processors()

        if 'module_ckpt' in self.load_model_kwargs:
            module_ckpt = self.load_model_kwargs.pop('module_ckpt')
            module_ckpt = torch.load(module_ckpt, map_location='cpu')['state_dict']
            self.load_state_dict(module_ckpt, strict=False)

    def encode_hidden_states(self, batch):
        image_embedding = super().encode_hidden_states(batch)

        action_emb = self.condition_transformer.batch_forward(batch, return_loss=False)[:, -1]
        return TupleTensor([image_embedding, action_emb])


def log_normal(size: tuple[int], mean: float, std: float, device: torch.device,
               dtype: torch.dtype = None) -> torch.Tensor:
    """
    Log normal distribution
    :param size: size of the tensor
    :param mean: mean of the distribution
    :param std: std of the distribution
    :param device: device of the tensor
    :param dtype: dtype of the Tensor
    :return: Tensor[size]
    """
    return torch.normal(mean, std, size=size, device=device, dtype=dtype).exp()


def image2condition_latent(image: torch.Tensor, vae: AutoencoderKLTemporalDecoder, mean: float,
                           std: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert image to condition latent
    :param image: Tensor[b c h w]
    :param vae: AutoencoderKLTemporalDecoder
    :param mean: mean of the noise
    :param std: std of the noise
    :return: condition_latent: Tensor[b c' h' w'] and condition_sigma: Tensor[b 1 1 1]
    """
    b, c, h, w = image.shape
    noise_aug_strength = log_normal(size=(b, 1, 1, 1), mean=mean, std=std, device=image.device, dtype=image.dtype)
    condition_image = torch.randn_like(image) * noise_aug_strength + image
    condition_latent = tensor2latent(condition_image[:, None, ...], vae)
    condition_latent /= vae.config.scaling_factor

    return condition_latent[:, 0, ...], noise_aug_strength


def get_add_time_ids(
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: torch.Tensor = 0.02,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
):
    add_time_ids = [[fps, motion_bucket_id, strength] for strength in noise_aug_strength]
    add_time_ids = torch.tensor(add_time_ids, dtype=dtype, device=device)
    return add_time_ids
