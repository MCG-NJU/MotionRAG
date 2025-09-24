from typing import Any, Optional

import torch
from torch.nn.functional import cosine_similarity
from torchmetrics import Metric
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Resize, Compose, CenterCrop, ToDtype, Normalize
from transformers import AutoModel


class DINOScore(Metric):
    """
    DINOScore metric for video-to-video
    """
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True
    full_state_update: Optional[bool] = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 100.0

    feature_network: str = "model"

    def __init__(self, model_path: str = 'facebook/dinov2-large', **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.model = AutoModel.from_pretrained(model_path)

        self.transform = Compose([
            Resize(256, InterpolationMode.BICUBIC, antialias=True),
            CenterCrop((224, 224)),
            ToDtype(torch.float16, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.add_state("dino_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    @torch.no_grad()
    @torch.amp.autocast('cuda')
    def _calc_similarity(self, video1: torch.Tensor, video2: torch.Tensor) -> torch.Tensor:
        video1 = self.transform(video1)
        video2 = self.transform(video2)

        video1_features = self.model(video1).last_hidden_state[:, 0]
        video2_features = self.model(video2).last_hidden_state[:, 0]

        return cosine_similarity(video1_features, video2_features, dim=-1).mean() * 100

    def update(self, videos1: torch.Tensor, videos2: torch.Tensor) -> None:
        assert len(videos1.shape) == 5, "videos1 must be of shape (B, T, C, H, W)"
        assert len(videos2.shape) == 5, "videos2 must be of shape (B, T, C, H, W)"

        for video1, video2 in zip(videos1, videos2):
            self.dino_score += self._calc_similarity(video1, video2)
            self.num_samples += 1

    def compute(self) -> torch.Tensor:
        return self.dino_score / self.num_samples


if __name__ == '__main__':
    c = DINOScore().cuda()
    videos1 = torch.randint(0, 255, (10, 16, 3, 256, 384), dtype=torch.uint8, device='cuda')
    videos2 = torch.randint(0, 255, (10, 16, 3, 256, 384), dtype=torch.uint8, device='cuda')
    c.update(videos1, videos2)
    print(c.compute())
