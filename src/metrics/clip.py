from typing import Any, Literal, Optional

import torch
from torchmetrics import Metric
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Resize, Compose, CenterCrop, ToDtype, Normalize


class CLIPScore(Metric):
    """
    CLIPScore metric for text-to-video and video-to-video
    """
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True
    full_state_update: Optional[bool] = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 100.0

    feature_network: str = "model"

    def __init__(self, mode: Literal['t2v', 'v2v'] = 't2v', **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.model = None
        self.tokenizer = None
        self.transform = None
        self.mode = mode

        self.configure_model()

        self.add_state("clip_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def configure_model(self):
        import open_clip
        self.model, _, _ = open_clip.create_model_and_transforms(
            'hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k', precision='fp16')
        self.tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')

        self.transform = Compose([
            Resize(224, InterpolationMode.BILINEAR, antialias=True),
            CenterCrop((224, 224)),
            ToDtype(torch.float16, scale=True),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

    @torch.no_grad()
    @torch.amp.autocast('cuda')
    def _calc_t2v_score(self, video: torch.Tensor, text: str | list[str]) -> torch.Tensor:
        text = self.tokenizer(text).to(self.device)
        video = self.transform(video)

        video_features = self.model.encode_image(video, normalize=True)
        text_features = self.model.encode_text(text, normalize=True)

        return (100.0 * (video_features * text_features).sum(dim=-1)).mean()

    @torch.no_grad()
    @torch.amp.autocast('cuda')
    def _calc_v2v_score(self, video1: torch.Tensor, video2: torch.Tensor) -> torch.Tensor:
        video1 = self.transform(video1)
        video2 = self.transform(video2)

        video1_features = self.model.encode_image(video1, normalize=True)
        video2_features = self.model.encode_image(video2, normalize=True)

        return (100.0 * (video1_features * video2_features).sum(dim=-1)).mean()

    def _update_t2v(self, videos: torch.Tensor, text: list[str]) -> None:
        assert len(videos.shape) == 5, "videos must be of shape (B, T, C, H, W)"
        assert len(text) == videos.shape[0], "text must be of same number of samples as videos"

        for video, t in zip(videos, text):
            self.clip_score += self._calc_t2v_score(video, t)
            self.num_samples += 1

    def _update_v2v(self, videos1: torch.Tensor, videos2: torch.Tensor) -> None:
        assert len(videos1.shape) == 5, "videos1 must be of shape (B, T, C, H, W)"
        assert len(videos2.shape) == 5, "videos2 must be of shape (B, T, C, H, W)"

        for video1, video2 in zip(videos1, videos2):
            self.clip_score += self._calc_v2v_score(video1, video2)
            self.num_samples += 1

    def update(self, *args, **kwargs) -> None:
        if self.mode == 't2v':
            self._update_t2v(*args, **kwargs)
        elif self.mode == 'v2v':
            self._update_v2v(*args, **kwargs)
        else:
            raise ValueError(f'Invalid mode: {self.mode}')

    def compute(self) -> torch.Tensor:
        return self.clip_score / self.num_samples


if __name__ == '__main__':
    c = CLIPScore(mode='t2v').cuda()
    videos = torch.randint(0, 255, (10, 16, 3, 256, 384), dtype=torch.uint8, device='cuda')
    text = ['a video of a cat', 'a video of a dog'] * 5
    c.update(videos, text)
    print(c.compute())

    c = CLIPScore(mode='v2v').cuda()
    videos1 = torch.randint(0, 255, (10, 16, 3, 256, 384), dtype=torch.uint8, device='cuda')
    videos2 = torch.randint(0, 255, (10, 16, 3, 256, 384), dtype=torch.uint8, device='cuda')
    c.update(videos1, videos2)
    print(c.compute())
