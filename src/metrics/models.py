import torch
from huggingface_hub import hf_hub_download
from torch.nn import Module
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Compose, Resize, CenterCrop, ToDtype, Normalize
from transformers import VideoMAEForVideoClassification
from transformers import AutoModel, AutoConfig


class VideoMAE(Module):
    def __init__(self, model_name: str = "MCG-NJU/videomae-base-finetuned-ssv2") -> None:
        super().__init__()
        self.model = VideoMAEForVideoClassification.from_pretrained(model_name)
        self.transform = Compose([
            Resize(224, InterpolationMode.BILINEAR, antialias=True),
            CenterCrop((224, 224)),
            ToDtype(torch.float16, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    @torch.amp.autocast('cuda')
    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        assert len(videos.shape) == 5, "Input must be of shape (B, T, C, H, W)"
        # uniform sample 16 frames
        videos = videos[:, torch.linspace(0, videos.shape[1] - 1, 16).round().long()]
        return self.model(pixel_values=self.transform(videos)).logits


class VideoMAE2(Module):
    def __init__(self, model_path: str = "OpenGVLab/VideoMAEv2-Large"):
        super().__init__()
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, config=config, trust_remote_code=True)

        self.transform = Compose([
            Resize(224, InterpolationMode.BILINEAR, antialias=True),
            CenterCrop((224, 224)),
            ToDtype(torch.float16, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    @torch.amp.autocast('cuda')
    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        assert len(videos.shape) == 5, "Input must be of shape (B, T, C, H, W)"
        # uniform sample 16 frames
        videos = videos[:, torch.linspace(0, videos.shape[1] - 1, 16).round().long()]
        pixel_values = self.transform(videos)
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4).contiguous()
        return self.model.forward(pixel_values)


class I3D(Module):
    def __init__(self, model_id: str = "flateon/FVD-I3D-torchscript", rescale=True, resize=True,
                 return_features=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        model_path = hf_hub_download(model_id, filename="i3d_torchscript.pt")
        self.model = torch.jit.load(model_path)
        self.num_features = 400
        self.rescale = rescale
        self.resize = resize
        self.return_features = return_features

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return self.model(x, rescale=self.rescale, resize=self.resize, return_features=self.return_features)
