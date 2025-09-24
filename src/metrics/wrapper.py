from typing import Any

import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


class SamplewiseWrapper(Metric):
    """
    This wrapper is used to calculate the sample-wise metrics (one value per sample)

    Args:
        metric: base metric that should be wrapped. It is assumed that the metric outputs a single
            tensor that is split along the first dimension.
    """

    def __init__(self, metric: Metric, **kwargs: Any):
        super().__init__(**kwargs)
        if not isinstance(metric, Metric):
            raise ValueError(f"Expected argument `metric` to be of type `Metric`, but got {type(metric)}")
        self.metric = metric

        self.add_state("score", default=[], dist_reduce_fx="cat")
        self.add_state("video_id", default=[], dist_reduce_fx="cat")

    def update(self, *args: Any, **kwargs: Any) -> None:
        video_id: None | torch.Tensor = kwargs.pop("video_id", None)

        score = self.metric(*args, **kwargs).view(-1)
        self.score.append(score)

        if video_id is not None:
            assert len(video_id) == len(score), \
                f"video_id must have the same length as score, but got {len(video_id)} and {len(score)}"
            self.video_id.append(video_id)

    def compute(self) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        score = dim_zero_cat(self.score)

        if len(self.video_id) == 0:
            # return in original order
            return score
        else:
            # dedup video_id and sort by video_id
            video_id = dim_zero_cat(self.video_id)
            assert len(score) == len(video_id), \
                f"score and video_id must have the same length, but got {len(score)} and {len(video_id)}"
            assert video_id.dtype in (torch.int64, torch.int32), \
                f"video_id must be int64 or int32, but got {video_id.dtype}"

            dedup_score = {}
            for b_idx, s in zip(video_id.tolist(), score):
                dedup_score[b_idx] = s

            return (torch.tensor([dedup_score[idx] for idx in sorted(dedup_score.keys())], device=score.device),
                    torch.tensor(sorted(dedup_score.keys()), device=score.device))

    def reset(self) -> None:
        """Reset metric."""
        super().reset()
        self.metric.reset()
