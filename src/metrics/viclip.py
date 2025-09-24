from typing import Literal, Any

import numpy as np
import torch
from transformers import AutoModel

from .clip import CLIPScore
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Resize, Compose, CenterCrop, ToDtype, Normalize, Lambda


class ViCLIPScore(CLIPScore):
    def __init__(self,
                 model_path: str = 'OpenGVLab/ViCLIP-L-14-hf',
                 mode: Literal['t2v', 'v2v'] = 't2v',
                 **kwargs: Any) -> None:
        self.model_path = model_path
        super().__init__(mode=mode, **kwargs)

    def configure_model(self):
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = self.model.tokenizer

        self.transform = Compose([
            Lambda(lambda x: x[:, np.linspace(0, x.size(1) - 1, 8).round()]),  # uniform_sample 8 frames
            Resize(224, InterpolationMode.BICUBIC, antialias=True),
            CenterCrop((224, 224)),
            ToDtype(torch.float16, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    @torch.amp.autocast('cuda')
    def _calc_t2v_score(self, video: torch.Tensor, text: str | list[str]) -> torch.Tensor:
        video = self.transform(video)

        video_features = self.model.encode_image(video, normalize=True)
        text_features = self.model.encode_text(text, normalize=True)

        return (100.0 * (video_features * text_features).sum(dim=-1)).mean()

    def _update_t2v(self, videos: torch.Tensor, text: list[str]) -> None:
        assert len(videos.shape) == 5, "videos must be of shape (B, T, C, H, W)"
        assert len(text) == videos.shape[0], "text must be of same number of samples as videos"

        self.clip_score += self._calc_t2v_score(videos, text)
        self.num_samples += videos.size(0)

    def _update_v2v(self, videos1: torch.Tensor, videos2: torch.Tensor) -> None:
        assert len(videos1.shape) == 5, "videos1 must be of shape (B, T, C, H, W)"
        assert len(videos2.shape) == 5, "videos2 must be of shape (B, T, C, H, W)"

        self.clip_score += self._calc_v2v_score(videos1, videos2)
        self.num_samples += videos1.size(0)


if __name__ == '__main__':
    c = ViCLIPScore(mode='t2v').cuda()
    videos = torch.randint(0, 255, (10, 16, 3, 256, 384), dtype=torch.uint8, device='cuda')
    text = ['a video of a cat', 'a video of a dog'] * 5
    c.update(videos, text)
    print(c.compute())

    c = ViCLIPScore(mode='v2v').cuda()
    videos1 = torch.randint(0, 255, (10, 16, 3, 256, 384), dtype=torch.uint8, device='cuda')
    videos2 = torch.randint(0, 255, (10, 16, 3, 256, 384), dtype=torch.uint8, device='cuda')
    c.update(videos1, videos2)
    print(c.compute())
