
from pathlib import Path

import torch
from lightning import pytorch as pl
from lightning.pytorch.utilities.memory import garbage_collection_cuda
from torchmetrics import Metric
from torchmetrics.image import FrechetInceptionDistance

from .action import I3DActionScore, MAEActionScore, MAE2ActionScore
from .clip import CLIPScore
from .viclip import ViCLIPScore
from .dino import DINOScore
from .fvd import FrechetVideoDistance
from .motion import MotionDistance
from .wrapper import SamplewiseWrapper


class MetricLogger(pl.Callback):
    def __init__(self, metric_name: str, metric: Metric, samplewise_metric: bool = False):
        self.metric_name = metric_name
        self.metric = metric
        self.samplewise_metric = samplewise_metric
        if self.samplewise_metric:
            self.metric = SamplewiseWrapper(self.metric)

    def calc(self, trainer, pl_module):
        generated_videos = [v['video'] for v in pl_module.generated_videos]
        gt_videos = [v['gt_video'] for v in pl_module.generated_videos]
        video_id = torch.tensor([v['id'] for v in pl_module.generated_videos]).split(1)

        assert len(generated_videos) == len(gt_videos), "Number of videos and gt must be equal"

        if self.samplewise_metric:
            for idx, pred, gt in zip(video_id, generated_videos, gt_videos):
                self.metric.update(pred.to(pl_module.device), gt.to(pl_module.device),
                                   video_id=idx.to(pl_module.device))
        else:
            for pred, gt in zip(generated_videos, gt_videos):
                self.metric.update(pred.to(pl_module.device), gt.to(pl_module.device))

        return self.metric.compute()

    def log_metric(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", prefix: str = "") -> None:
        trainer.strategy.barrier()
        trainer.print(f"Calculating {self.metric_name}...")

        self.metric.reset()
        self.metric = self.metric.to(pl_module.device)
        try:
            if self.samplewise_metric:
                metric_value, video_id = self.calc(trainer, pl_module)
                if hasattr(pl_module, 'sample_metrics'):
                    pl_module.sample_metrics.update({self.metric_name: metric_value.cpu()})
                    pl_module.sample_metrics.update({'id': video_id.cpu()})
            else:
                metric_value = self.calc(trainer, pl_module)
        except Exception as e:
            metric_value = torch.tensor([torch.nan], device=pl_module.device)
            trainer.print(f'{self.metric_name} failed: {e}')
        self.metric = self.metric.to("cpu")

        pl_module.log(f"{prefix}/{self.metric_name}", metric_value.mean(), sync_dist=True)

        garbage_collection_cuda()

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if hasattr(pl_module, 'on_validation_epoch_end'):
            pl_module.on_validation_epoch_end()  # A hack to execute the pl_module's callback first.

        self.log_metric(trainer, pl_module, prefix="val")

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if hasattr(pl_module, 'on_test_epoch_end'):
            pl_module.on_test_epoch_end()

        self.log_metric(trainer, pl_module, prefix="test")


class SaveSampleMetrics(pl.Callback):
    """Save metrics of each sample to disk"""

    def __init__(self, save_path: str | Path = 'videos', create_subdir: bool = True):
        self.save_path = Path(save_path)
        self.create_subdir = create_subdir
        self.annotations = None

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.is_global_zero:
            pl_module.sample_metrics = {}
            if hasattr(trainer.val_dataloaders.dataset, 'annotations'):
                self.annotations = trainer.val_dataloaders.dataset.annotations

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.is_global_zero:
            pl_module.sample_metrics = {}
            if hasattr(trainer.test_dataloaders.dataset, 'annotations'):
                self.annotations = trainer.test_dataloaders.dataset.annotations

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        trainer.strategy.barrier()
        if trainer.is_global_zero and len(pl_module.sample_metrics) != 0:
            sample_metrics = {m: v.tolist() for m, v in pl_module.sample_metrics.items()}
            length_list = [len(v) for v in pl_module.sample_metrics.values()]
            assert len(set(length_list)) == 1, f'Length of sample metrics is not equal: {length_list}'

            if self.annotations is not None:
                try:
                    id2anno = {anno['id']: anno for anno in self.annotations}
                    sample_metrics['video'] = [id2anno[idx]['video'] for idx in sample_metrics['id']]
                    sample_metrics['start_sec'] = [id2anno[idx]['start_sec'] for idx in sample_metrics['id']]
                    sample_metrics['end_sec'] = [id2anno[idx]['end_sec'] for idx in sample_metrics['id']]
                except KeyError:
                    pass

            # dict of tensor to list of values
            sample_info: list[dict] = [dict(zip(sample_metrics, t)) for t in zip(*sample_metrics.values())]

            if self.create_subdir:
                path = self.save_path / str(trainer.logger.version) / str(trainer.global_step) / f'sample_metrics.pt'
            else:
                path = self.save_path / f'sample_metrics.pt'

            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(sample_info, path)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.on_validation_end(trainer, pl_module)


class FVDMetric(MetricLogger):
    """Calc the FVD metric"""

    def __init__(self, model_path: str = 'flateon/FVD-I3D-torchscript'):
        self.model_path = model_path
        super().__init__('fvd', metric=FrechetVideoDistance(self.model_path))
        self.metric.update_ = self.metric.update
        self.metric.update = self.update

    def update(self, preds, target):
        self.metric.update_(preds, real=False)
        self.metric.update_(target, real=True)


class FIDMetric(MetricLogger):
    """Calc the FID metric"""

    def __init__(self):
        super().__init__('fid', metric=FrechetInceptionDistance())
        self.metric.update_ = self.metric.update
        self.metric.update = self.update

    def update(self, preds, target):
        self.metric.update_(preds.flatten(0, 1), real=False)
        self.metric.update_(target.flatten(0, 1), real=True)


class MotionMetric(MetricLogger):
    """Calc the motion metric"""

    def __init__(self):
        super().__init__('motion',
                         metric=MotionDistance(),
                         samplewise_metric=True)


class ActionMetric(MetricLogger):
    """Calc the action metric"""

    def __init__(self, model_path: str = 'flateon/FVD-I3D-torchscript'):
        self.model_path = model_path
        super().__init__('action',
                         metric=I3DActionScore(model_path=model_path),
                         samplewise_metric=True)


class MAEActionMetric(MetricLogger):
    """Calc the action metric"""

    def __init__(self, model_name='MCG-NJU/videomae-base-finetuned-ssv2'):
        self.model_name = model_name
        super().__init__('action_mae',
                         metric=MAEActionScore(self.model_name),
                         samplewise_metric=True)


class MAE2ActionMetric(MetricLogger):
    """Calc the action metric"""

    def __init__(self, model_path="OpenGVLab/VideoMAEv2-Large"):
        self.model_path = model_path
        super().__init__('action_mae2',
                         metric=MAE2ActionScore(self.model_path),
                         samplewise_metric=True)


class ClipT2VMetric(MetricLogger):
    """Calc the clip t2v metric"""

    def __init__(self):
        super().__init__('clip_t2v',
                         metric=CLIPScore(mode='t2v'),
                         samplewise_metric=True)

    def calc(self, trainer, pl_module):
        generated_videos = [v['video'] for v in pl_module.generated_videos]
        text = [[v['prompt']] for v in pl_module.generated_videos]
        video_id = torch.tensor([v['id'] for v in pl_module.generated_videos]).split(1)

        assert len(generated_videos) == len(text), "Number of videos and text must be equal"
        clip = self.metric
        for idx, video, t in zip(video_id, generated_videos, text):
            clip.update(video.to(pl_module.device), t, video_id=idx.to(pl_module.device))

        return clip.compute()


class ClipV2VMetric(MetricLogger):
    """Calc the clip v2v metric"""

    def __init__(self):
        super().__init__('clip_v2v',
                         CLIPScore(mode='v2v'),
                         samplewise_metric=True)


class ViClipT2VMetric(ClipT2VMetric):
    """Calc the Viclip t2v metric"""

    def __init__(self):
        MetricLogger.__init__(self,
                              'Viclip_t2v',
                              metric=ViCLIPScore(mode='t2v'),
                              samplewise_metric=True)


class ViClipV2VMetric(MetricLogger):
    """Calc the Viclip v2v metric"""

    def __init__(self):
        super().__init__('Viclip_v2v',
                         metric=ViCLIPScore(mode='v2v'),
                         samplewise_metric=True)


class DINOMetric(MetricLogger):
    """Calc the DINO metric"""

    def __init__(self, model_path: str = 'facebook/dinov2-large'):
        super().__init__('dino',
                         metric=DINOScore(model_path=model_path),
                         samplewise_metric=True)
