import torch
import torch.nn as nn
import kornia
import open_clip
from torch.utils.checkpoint import checkpoint
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Compose, Resize, CenterCrop, Normalize
from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel, VideoMAEModel, AutoConfig, AutoModel


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):
    def encode(self, x):
        return x


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class', ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def forward(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0. and not disable_dropout:
            mask = 1. - torch.bernoulli(torch.ones_like(c) * self.ucg_rate)
            c = mask * c + (1 - mask) * torch.ones_like(c) * (self.n_classes - 1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = torch.ones((bs,), device=device) * uc_class
        uc = {self.key: uc}
        return uc


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class FrozenT5Embedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""

    def __init__(self, version="google/t5-v1_1-large", device="cuda", max_length=77,
                 freeze=True):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length  # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        # self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]

    def __init__(self, version="openai/clip-vit-large-patch14", device="cpu", max_length=77,
                 freeze=True, layer="last", layer_idx=None):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        self.dim = self.transformer.config.hidden_size
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        # self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer == "hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        # "pooled",
        "last",
        "penultimate"
    ]

    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.dim = model.ln_final.weight.shape[0]
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text, return_cls_tokens=False):
        tokens = open_clip.tokenize(text)  ## all clip models use 77 as context length
        z = self.encode_with_transformer(tokens.to(self.device))
        if return_cls_tokens:
            return z[torch.arange(len(text)), tokens.argmax(-1)], z
        else:
            return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPImageEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """

    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77,
                 freeze=True, layer="pooled", antialias=True, ucg_rate=0.):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'),
                                                            pretrained=version, )
        del model.transformer
        self.model = model
        # self.mapper = torch.nn.Linear(1280, 1024)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "penultimate":
            raise NotImplementedError()
            self.layer_idx = 1

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)
        self.ucg_rate = ucg_rate

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic', align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.cuda.amp.autocast()
    def forward(self, image, no_dropout=False):
        z = self.encode_with_vision_transformer(image)
        if self.ucg_rate > 0. and not no_dropout:
            z = torch.bernoulli((1. - self.ucg_rate) * torch.ones(z.shape[0], device=z.device))[:, None] * z
        return z

    def encode_with_vision_transformer(self, img):
        img = self.preprocess(img)
        x = self.model.visual(img)
        return x

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPImageEmbedderV2(AbstractEncoder):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """

    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda",
                 freeze=True, layer="pooled", antialias=True):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'),
                                                            pretrained=version, )
        del model.transformer
        self.model = model
        self.device = device

        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "penultimate":
            raise NotImplementedError()
            self.layer_idx = 1

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic', align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image, no_dropout=False):
        ## image: b c h w
        z = self.encode_with_vision_transformer(image)
        return z

    def encode_with_vision_transformer(self, x):
        x = self.preprocess(x)

        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        if self.model.visual.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(x.shape[0], x.shape[1], self.model.visual.grid_size[0], self.model.visual.patch_size[0],
                          self.model.visual.grid_size[1], self.model.visual.patch_size[1])
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.model.visual.grid_size[0] * self.model.visual.grid_size[1], -1)
            x = self.model.visual.patchnorm_pre_ln(x)
            x = self.model.visual.conv1(x)
        else:
            x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [self.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                         device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.visual.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.model.visual.patch_dropout(x)
        x = self.model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        return x


class FrozenCLIPT5Encoder(AbstractEncoder):
    def __init__(self, clip_version="openai/clip-vit-large-patch14", t5_version="google/t5-v1_1-xl", device="cuda",
                 clip_max_length=77, t5_max_length=77):
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(clip_version, device, max_length=clip_max_length)
        self.t5_encoder = FrozenT5Embedder(t5_version, device, max_length=t5_max_length)

    def encode(self, text):
        return self(text)

    def forward(self, text):
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]


class VideoMAEEmbedder(AbstractEncoder):
    """
    Uses the VideoMAE vision transformer encoder for videos
    """

    def __init__(self, model: str = "MCG-NJU/videomae-base-finetuned-ssv2", freeze: bool = True, compile: bool = False):
        super().__init__()
        self.model = VideoMAEModel.from_pretrained(model)
        self.transform = Compose([
            Resize(224, InterpolationMode.BILINEAR, antialias=True),
            CenterCrop((224, 224)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.dim = self.model.config.hidden_size

        if freeze:
            self.freeze()
        if compile:
            self.model = torch.compile(self.model, fullgraph=True)

    def preprocess(self, x):
        # normalize to [0,1]
        x = (x + 1.) / 2.
        x = self.transform(x)
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, video):
        ## video: b t c h w
        assert len(video.shape) == 5, "Input must be of shape (B, T, C, H, W)"

        # uniform sampling 16 frames
        video = video[:, torch.linspace(0, video.shape[1] - 1, 16).round().long()]

        pixel_values = self.preprocess(video)
        last_hidden_state = self.model(pixel_values=pixel_values).last_hidden_state
        return last_hidden_state


class VideoMAE2Embedder(AbstractEncoder):
    """
    Uses the VideoMAE2Embedder vision transformer encoder for videos
    """

    def __init__(self, model_path: str = "OpenGVLab/VideoMAEv2-Large", freeze=True):
        super().__init__()
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)

        self.transform = Compose([
            Resize(224, InterpolationMode.BILINEAR, antialias=True),
            CenterCrop((224, 224)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.dim = config.model_config['embed_dim']

        if freeze:
            self.freeze()

    def preprocess(self, x):
        # normalize to [0,1]
        x = (x + 1.) / 2.
        x = self.transform(x)
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward_features(self, x):
        self = self.model.model
        B = x.size(0)

        x = self.patch_embed(x)

        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(
                x.device).clone().detach()
        x = self.pos_drop(x)

        for blk in self.blocks:
            if self.with_cp:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x

    def forward(self, video):
        ## video: b t c h w
        assert len(video.shape) == 5, "Input must be of shape (B, T, C, H, W)"

        # uniform sampling 16 frames
        video = video[:, torch.linspace(0, video.shape[1] - 1, 16).round().long()]

        pixel_values = self.preprocess(video)
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4).contiguous()
        last_hidden_state = self.forward_features(pixel_values)
        return last_hidden_state


class CLIPImageEmbedder(AbstractEncoder):
    """
    Uses the CLIP image encoder for images
    """

    def __init__(
            self,
            model: str = "openai/clip-vit-large-patch14",
            freeze=True,
            dtype: torch.dtype = torch.float16,
            compile: bool = False,
            model_kwargs=None,
    ):
        super().__init__()
        model_kwargs = dict() if model_kwargs is None else model_kwargs

        from transformers import CLIPVisionModelWithProjection
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            model,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
            **model_kwargs,
        )

        self.image_size = self.model.config.image_size
        self.dim = self.model.vision_model.config.hidden_size

        self.transform = Compose([
            Resize(self.image_size, InterpolationMode.BICUBIC, antialias=True),
            CenterCrop((self.image_size, self.image_size)),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

        if freeze:
            self.freeze()
        if compile:
            self.model = torch.compile(self.model, fullgraph=True)

    def preprocess(self, x):
        # normalize to [0,1]
        x = (x + 1.) / 2.
        x = self.transform(x)
        return x

    def freeze(self):
        self.model = self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def get_uncond_emb(self):
        uncond_image = torch.zeros_like(self.transform(torch.zeros(1, 3, 224, 224, device=self.model.device)))
        return self.model(uncond_image, output_hidden_states=True).hidden_states[-2]

    def forward(self, images):
        assert len(images.shape) == 4, "Input must be of shape (B, C, H, W)"

        pixel_values = self.preprocess(images)
        last_hidden_state = self.model(pixel_values, output_hidden_states=True).hidden_states[-2]
        return last_hidden_state


class SDXLImageEmbedder(CLIPImageEmbedder):
    """
    Uses the SDXL IP adapter plus image transformer encoder for images
    """

    def __init__(self, freeze=True, dtype: torch.dtype = torch.float16, compile: bool = False):
        super().__init__(
            model="h94/IP-Adapter",
            freeze=freeze,
            dtype=dtype,
            compile=compile,
            model_kwargs=dict(
                subfolder="models/image_encoder"
            ),
        )


class KolorsImageEmbedder(CLIPImageEmbedder):
    """
    Uses the Kolors image transformer encoder for images
    """

    def __init__(self, freeze=True, dtype=torch.float16, compile: bool = False):
        super().__init__(
            model="Kwai-Kolors/Kolors-IP-Adapter-Plus",
            freeze=freeze,
            dtype=dtype,
            compile=compile,
            model_kwargs=dict(
                subfolder="image_encoder",
                revision="refs/pr/4",
            ),
        )


class DINOImageEmbedder(CLIPImageEmbedder):
    """
    Uses the DINO image encoder for images
    """

    def __init__(
            self,
            model: str = "facebook/dinov2-large",
            freeze=True,
            dtype: torch.dtype = torch.bfloat16,
            compile: bool = False,
            model_kwargs=None,
    ):
        super().__init__()
        model_kwargs = dict() if model_kwargs is None else model_kwargs

        from transformers import AutoModel, AutoProcessor
        self.model = AutoModel.from_pretrained(
            model,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
            **model_kwargs,
        )
        preprocessor = AutoProcessor.from_pretrained(model)

        self.dim = self.model.config.hidden_size

        self.transform = Compose([
            Resize(preprocessor.size['shortest_edge'], InterpolationMode.BICUBIC, antialias=True),
            CenterCrop((preprocessor.crop_size['height'], preprocessor.crop_size['width'])),
            Normalize(mean=preprocessor.image_mean, std=preprocessor.image_std)
        ])

        if freeze:
            self.freeze()
        if compile:
            self.model = torch.compile(self.model, fullgraph=True)

    def forward(self, images):
        assert len(images.shape) == 4, "Input must be of shape (B, C, H, W)"

        pixel_values = self.preprocess(images)
        last_hidden_state = self.model(pixel_values).last_hidden_state
        return last_hidden_state
