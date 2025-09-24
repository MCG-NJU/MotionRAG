from typing import Any, Optional

import torch
from torchmetrics import Metric
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.transforms import v2 as tvtf


class MotionDistance(Metric):
    """
    Calculate the motion distance between two videos.
    """
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = False
    full_state_update: Optional[bool] = False
    plot_lower_bound: float = 0.0

    feature_network: str = "model"

    def __init__(self, hist_bins: int = 256, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        self.hist_bins = hist_bins
        self.transform = tvtf.Compose([
            tvtf.Resize(256, interpolation=tvtf.InterpolationMode.BILINEAR, antialias=True),
            tvtf.ToDtype(torch.float16, scale=True),
            tvtf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.add_state("motion_kl", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    @torch.no_grad()
    def calc_motion_hist(self, video) -> torch.Tensor:
        """
        Calculate motion histogram for each frame in the video.
        """
        video = self.transform(video)
        h, w = video.shape[-2:]
        if h % 8 != 0 or w % 8 != 0:
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            video = tvtf.Pad(padding=(0, 0, pad_w, pad_h), padding_mode="reflect")(video)

        with torch.amp.autocast('cuda'):
            frame1, frame2 = video[:-1], video[1:]
            flow = self.model(frame1, frame2)[-1]

        flow_mag = torch.linalg.vector_norm(flow, dim=1)
        log_flow_mag = torch.log2_(flow_mag).cpu()
        flow_mag_feat = torch.stack(
            [torch.histc(log_f, bins=self.hist_bins, min=-7, max=5) for log_f in log_flow_mag])

        flow_mag_feat += 0.1
        flow_mag_feat /= flow_mag_feat.sum(dim=1, keepdim=True)

        return flow_mag_feat

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        assert preds.shape == target.shape, f"preds and target must have the same shape, but got {preds.shape} and {target.shape}"
        assert len(preds.shape) == 5, f"preds and target must have 5 dimensions, but got {len(preds.shape)}"

        for pred, target in zip(preds, target):
            pred_motion = self.calc_motion_hist(pred)
            target_motion = self.calc_motion_hist(target)

            motion_kl = (target_motion * (target_motion.log() - pred_motion.log())).sum(dim=1)
            self.motion_kl += motion_kl.mean()
            self.num_samples += 1

    def compute(self) -> torch.Tensor:
        return self.motion_kl / self.num_samples


if __name__ == '__main__':
    c = MotionDistance().cuda()
    videos1 = torch.randint(0, 255, (10, 16, 3, 256, 384), dtype=torch.uint8, device='cuda')
    videos2 = torch.randint(0, 255, (10, 16, 3, 256, 384), dtype=torch.uint8, device='cuda')
    c.update(videos1, videos2)
    print(c.compute())