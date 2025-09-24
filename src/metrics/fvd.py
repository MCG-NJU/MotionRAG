from typing import Any

import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from .models import I3D


class FrechetVideoDistance(FrechetInceptionDistance):
    def __init__(
            self,
            model_id: str = "flateon/FVD-I3D-torchscript",
            reset_real_features: bool = True,
            normalize: bool = False,
            **kwargs: Any,
    ) -> None:
        model = I3D(model_id)
        super().__init__(model, reset_real_features, normalize, **kwargs)


if __name__ == '__main__':
    c = FrechetVideoDistance().cuda()
    videos1 = torch.randint(0, 255, (10, 16, 3, 256, 384), dtype=torch.uint8, device='cuda')
    videos2 = torch.randint(0, 255, (10, 16, 3, 256, 384), dtype=torch.uint8, device='cuda')
    c.update(videos1, real=True)
    c.update(videos2, real=False)
    print(c.compute())

    c = FrechetInceptionDistance().cuda()
    c.update(videos1.flatten(0, 1), real=True)
    c.update(videos2.flatten(0, 1), real=False)
    print(c.compute())
