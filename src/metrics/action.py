from typing import Any, Optional

import torch
from torch.nn import Module
from torchmetrics import Metric

from .models import VideoMAE, I3D, VideoMAE2


class ActionScore(Metric):
    """
    Action Score metric for video generation.
    """
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = True
    full_state_update: Optional[bool] = False
    plot_lower_bound: float = 0.0
    plot_upper_bound: float = 100.0

    feature_network: str = "model"

    def __init__(self, model: Module, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model

        self.add_state("action_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        assert preds.shape == target.shape, f"preds and target must have the same shape, but got {preds.shape} and {target.shape}"
        assert len(preds.shape) == 5, f"preds and target must have 5 dimensions, but got {len(preds.shape)}"

        pred_feat = self.model(preds)
        target_feat = self.model(target)

        pred_feat /= torch.linalg.vector_norm(pred_feat, dim=1, keepdim=True)
        target_feat /= torch.linalg.vector_norm(target_feat, dim=1, keepdim=True)

        self.action_score += (100 * torch.sum(pred_feat * target_feat, dim=1)).sum()
        self.num_samples += len(preds)

    def compute(self) -> torch.Tensor:
        return self.action_score / self.num_samples


class I3DActionScore(ActionScore):
    def __init__(self, model_id: str = "flateon/FVD-I3D-torchscript", **kwargs) -> None:
        model = I3D(model_id)
        super().__init__(model, **kwargs)


class MAEActionScore(ActionScore):
    def __init__(self, model_name: str = "MCG-NJU/videomae-base-finetuned-ssv2", **kwargs) -> None:
        model = VideoMAE(model_name)
        super().__init__(model, **kwargs)


class MAE2ActionScore(ActionScore):
    def __init__(self, model_path: str = "OpenGVLab/VideoMAEv2-Large", **kwargs) -> None:
        model = VideoMAE2(model_path)
        super().__init__(model, **kwargs)


if __name__ == '__main__':
    c = I3DActionScore().cuda()
    videos1 = torch.randint(0, 255, (10, 16, 3, 256, 384), dtype=torch.uint8, device='cuda')
    videos2 = torch.randint(0, 255, (10, 16, 3, 256, 384), dtype=torch.uint8, device='cuda')
    c.update(videos1, videos2)
    print(c.compute())

    c = MAEActionScore().cuda()
    videos1 = torch.randint(0, 255, (10, 16, 3, 256, 384), dtype=torch.uint8, device='cuda')
    videos2 = torch.randint(0, 255, (10, 16, 3, 256, 384), dtype=torch.uint8, device='cuda')
    c.update(videos1, videos2)
    print(c.compute())

    c = MAE2ActionScore().cuda()
    videos1 = torch.randint(0, 255, (10, 16, 3, 256, 384), dtype=torch.uint8, device='cuda')
    videos2 = torch.randint(0, 255, (10, 16, 3, 256, 384), dtype=torch.uint8, device='cuda')
    c.update(videos1, videos2)
    print(c.compute())
