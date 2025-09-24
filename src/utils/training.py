import multiprocessing
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import lightning.pytorch as pl
import torch
from PIL import Image
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import grad_norm
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.optim import Optimizer
from torchvision.io import write_video

from src.utils.pipeline import denormalize


class DatasetTimer(pl.Callback):
    """Logs the time spent on reading video and transforms"""

    @staticmethod
    def time_logger(pl_module: "pl.LightningModule", batch: Any, batch_idx: int) -> None:
        metadata = batch['metadata']
        read_video_time = [m['read_video_time'] for m in metadata]
        transforms_time = [m['transforms_time'] for m in metadata]
        clip_length = [m['clip_length'] for m in metadata]

        pl_module.log("dataset/read_video_time", sum(read_video_time) / len(read_video_time), on_step=True)
        pl_module.log("dataset/transforms_time", sum(transforms_time) / len(transforms_time), on_step=True)
        pl_module.log("dataset/clip_length", sum(clip_length) / len(clip_length), on_step=True)

    def on_train_batch_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        self.time_logger(pl_module, batch, batch_idx)

    def on_validation_batch_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        self.time_logger(pl_module, batch, batch_idx)

    def on_test_batch_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        self.time_logger(pl_module, batch, batch_idx)

    def on_predict_batch_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        self.time_logger(pl_module, batch, batch_idx)


class IncrementalCheckpoint(pl.Callback):
    """Save checkpoint with only the incremental part of the model"""

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        pl_module.strict_loading = False

    def on_save_checkpoint(
            self, trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            checkpoint: Dict[str, Any]
    ) -> None:
        for key in list(checkpoint['state_dict'].keys()):
            should_save = False
            for module_name in pl_module.full_trainable_modules:
                if key.startswith(module_name):
                    should_save = True

            for module_name in pl_module.lora_trainable_modules:
                if key.startswith(module_name):
                    should_save = True

            if not should_save:
                del checkpoint['state_dict'][key]


class GradientMonitor(pl.Callback):
    """Logs the gradient norm"""

    def __init__(self, norm_type: int | str = 2):
        norm_type = float(norm_type)
        if norm_type <= 0:
            raise ValueError(f"`norm_type` must be a positive number or 'inf' (infinity norm). Got {norm_type}")
        self.norm_type = norm_type

    def on_before_optimizer_step(
            self, trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            optimizer: Optimizer
    ) -> None:
        norms = grad_norm(pl_module, norm_type=self.norm_type)
        max_grad = torch.tensor([v for k, v in norms.items() if k != f"grad_{self.norm_type}_norm_total"]).max()
        pl_module.log_dict({'train/grad/max': max_grad, 'train/grad/total': norms[f"grad_{self.norm_type}_norm_total"]})


class SaveVideo(pl.Callback):
    """Saves the video to disk"""

    def __init__(self,
                 type: Literal["generate", "gt"] = 'generate',
                 save_path: str = 'videos',
                 video_num: int = None,
                 save_mode: Literal['batch', 'immediate'] = 'batch',
                 create_subdir: bool = True,
                 suffix: str = 'mp4',
                 ):
        """
        Saves video to disk
        :param type: type of video to save, either 'generate' or 'gt'
        :param save_path: path to save the video
        :param create_subdir: whether to create a subdirectory for each run
        :param video_num: number of videos to save
        :param save_mode: strategy for saving videos, either 'batch' to save after all videos are generated, or 'immediate' to save each video as it is generated
        """
        self.type = type
        self.base_save_path = Path(save_path)
        self.save_path = None
        self.video_num = video_num
        self.create_subdir = create_subdir
        self.save_mode = save_mode
        self.suffix = suffix
        self.writer = partial(write_video, fps=8, video_codec='libvpx-vp9', options={'crf':      '27',
                                                                                     'cpu-used': '1',
                                                                                     'deadline': 'good'})

    def prepare_path(self, trainer: "pl.Trainer"):
        trainer.strategy.barrier()
        if self.create_subdir:
            uid = trainer.strategy.broadcast(str(trainer.logger.version), src=0)
            save_path = self.base_save_path / uid / str(trainer.global_step) / self.type
            save_path.mkdir(parents=True, exist_ok=True)
        else:
            save_path = self.base_save_path
            save_path.mkdir(parents=True, exist_ok=True)
        self.save_path = save_path

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.prepare_path(trainer)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.prepare_path(trainer)

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        if self.save_mode == 'batch':
            return

        pl_module.output_assertions(outputs, batch)

        metadata: list[dict] = batch['metadata']
        gt_videos = denormalize(batch['video']).cpu() if 'video' in batch else None

        for i, metadata_item in enumerate(metadata):
            save_name = metadata_item['save_name']

            if self.type == "generate":
                video = outputs[i]
            elif self.type == "gt":
                assert gt_videos is not None, "GT video not found"
                video = gt_videos[i]
            else:
                raise ValueError(f"Invalid type {self.type}")

            self.writer(str(self.save_path / f"{save_name}.{self.suffix}"), video.permute(0, 2, 3, 1))

    def on_test_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0
    ) -> None:
        self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.save_mode == 'immediate':
            return

        video_infos = pl_module.generated_videos

        if self.video_num is not None:
            video_infos = video_infos[:self.video_num // trainer.world_size]

        if self.type == "generate":
            write_args = [(str(self.save_path / f"{info['save_name']}.{self.suffix}"),
                           info['video'][0].permute(0, 2, 3, 1)) for info in video_infos]
        elif self.type == 'gt':
            assert all(info['gt_video'] is not None for info in video_infos), "GT video not found"
            write_args = [(str(self.save_path / f"{info['save_name']}.{self.suffix}"),
                           info['gt_video'][0].permute(0, 2, 3, 1)) for info in video_infos]
        else:
            raise ValueError(f"Invalid type {self.type}")

        trainer.print(f"Saving {self.type} videos")
        # parallelize video saving
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(32) as p:
            p.starmap(self.writer, write_args, chunksize=1)
        trainer.strategy.barrier()

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return self.on_validation_end(trainer, pl_module)


def write_image(path: str, image: Tensor) -> None:
    """
    Writes image to disk
    :param path: path to save the image
    :param image: Tensor [1 H W C] or [H W C]
    :return: None
    """
    assert len(image.shape) == 4 and image.size(0) == 1 or len(image.shape) == 3, "Image must be [1 H W C] or [H W C]"
    assert image.size(-1) == 3, "Image must have 3 channels"
    assert image.dtype == torch.uint8, "Image must be uint8"

    image = image.squeeze().cpu().numpy()
    Image.fromarray(image).save(path)


class SaveImage(SaveVideo):
    """Saves the image to disk"""

    def __init__(self,
                 type: Literal["generate", "gt"] = 'generate',
                 save_path: str = 'images',
                 image_num: int = None,
                 save_mode: Literal['batch', 'immediate'] = 'batch',
                 create_subdir: bool = True,
                 suffix: str = 'png'
                 ):
        super().__init__(type, save_path, image_num, save_mode, create_subdir, suffix)
        self.writer = write_image


class WandbVideoLogger(SaveVideo):
    """Logs the video to wandb"""

    def __init__(self, type: Literal["generate", "gt"] = 'generate', save_path: str = 'videos',
                 save_mode: Literal['batch', 'immediate'] = 'batch', remove_after_log: bool = False,
                 video_num: int = 40):
        """
        Logs the video to wandb
        :param type: type of video to save, either 'generate' or 'gt'
        :param save_path: path to save the video
        :param remove_after_log: remove the video from disk after logging
        :param video_num: number of videos to save
        """
        super().__init__(type, save_path, video_num, save_mode)
        self.remove_after_log = remove_after_log

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_validation_end(trainer, pl_module)

        if trainer.global_rank == 0 and isinstance(trainer.logger, WandbLogger):
            videos = sorted(self.save_path.rglob("*.mp4"))[:self.video_num]
            trainer.logger.log_video('val' if self.type == 'generate' else 'gt', [str(s) for s in videos],
                                     step=trainer.global_step)

            if self.remove_after_log:
                for video in videos:
                    video.unlink(missing_ok=True)
                self.save_path.rmdir()

        trainer.strategy.barrier()

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return self.on_validation_end(trainer, pl_module)


class WandbCodeLogger(pl.Callback):
    def __init__(self, code_path: str = 'src', code_suffix: str = '.py'):
        self.code_path = code_path
        self.code_suffix = code_suffix
        self.logged = False

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        if trainer.global_rank == 0 and isinstance(trainer.logger, WandbLogger) and not self.logged:
            trainer.logger.experiment.log_code(self.code_path, include_fn=lambda path: path.endswith(self.code_suffix))
            self.logged = True


class FnCallWarpper:
    def __init__(self, fn: str, from_class: str = None, args: List = None, kwargs: Dict = None, return_fn=False):
        if from_class is not None:
            class_module, from_class = from_class.rsplit(".", 1)
            class_module = __import__(class_module, fromlist=[from_class])
            from_class = getattr(class_module, from_class)
            self.fn = getattr(from_class, fn)
        else:
            module, fn = fn.rsplit(".", 1)
            module = __import__(module, fromlist=[fn])
            self.fn = getattr(module, fn)
        self.return_fn = return_fn
        self.args = args if args else []
        self.kwargs = kwargs if kwargs else {}
        if return_fn:
            self._result = self.fn
        else:
            self._result = self.fn(*self.args, **self.kwargs)

    def __call__(self, *args, **kwargs):
        if self.return_fn:
            return self._result(*args, **kwargs)
        return self._result

    def __code__(self):
        return self.fn.__code__


class FindUnusedParameters(pl.Callback):
    def on_before_optimizer_step(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
                                 optimizer: Optimizer) -> None:
        for name, p in pl_module.named_parameters():
            if p.requires_grad and p.grad is None:
                print(f'Unused parameter: {name}')


class CommandLineCallback(pl.Callback):
    def __init__(self, command: str, when: Literal[
        'on_init_start', 'on_init_end', 'on_fit_start', 'on_fit_end', 'on_train_batch_start', 'on_train_batch_end',
        'on_validation_batch_start', 'on_validation_batch_end', 'on_test_batch_start', 'on_test_batch_end',
        'on_predict_batch_start', 'on_predict_batch_end', 'on_epoch_start', 'on_epoch_end', 'on_train_start',
        'on_train_end', 'on_validation_start', 'on_validation_end', 'on_test_start', 'on_test_end', 'on_predict_start',
        'on_predict_end', 'init'], rank_zero_only: bool = False):
        self.when = when
        self.command = command
        self.rank_zero_only = rank_zero_only
        if when == 'init':
            self.global_rank = 0
            self.exec(self, None)

    def exec(self, trainer, pl_module):
        if self.rank_zero_only and trainer.global_rank != 0:
            return
        import subprocess
        result = subprocess.run(self.command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f'Command failed: {result.stderr}')

    def on_init_start(self, trainer, pl_module):
        if self.when == 'on_init_start':
            self.exec(trainer, pl_module)

    def on_init_end(self, trainer, pl_module):
        if self.when == 'on_init_end':
            self.exec(trainer, pl_module)

    def on_fit_start(self, trainer, pl_module):
        if self.when == 'on_fit_start':
            self.exec(trainer, pl_module)

    def on_fit_end(self, trainer, pl_module):
        if self.when == 'on_fit_end':
            self.exec(trainer, pl_module)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.when == 'on_train_batch_start':
            self.exec(trainer, pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.when == 'on_train_batch_end':
            self.exec(trainer, pl_module)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if self.when == 'on_validation_batch_start':
            self.exec(trainer, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.when == 'on_validation_batch_end':
            self.exec(trainer, pl_module)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if self.when == 'on_test_batch_start':
            self.exec(trainer, pl_module)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.when == 'on_test_batch_end':
            self.exec(trainer, pl_module)

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if self.when == 'on_predict_batch_start':
            self.exec(trainer, pl_module)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.when == 'on_predict_batch_end':
            self.exec(trainer, pl_module)

    def on_epoch_start(self, trainer, pl_module):
        if self.when == 'on_epoch_start':
            self.exec(trainer, pl_module)

    def on_epoch_end(self, trainer, pl_module):
        if self.when == 'on_epoch_end':
            self.exec(trainer, pl_module)

    def on_train_start(self, trainer, pl_module):
        if self.when == 'on_train_start':
            self.exec(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        if self.when == 'on_train_end':
            self.exec(trainer, pl_module)

    def on_validation_start(self, trainer, pl_module):
        if self.when == 'on_validation_start':
            self.exec(trainer, pl_module)

    def on_validation_end(self, trainer, pl_module):
        if self.when == 'on_validation_end':
            self.exec(trainer, pl_module)

    def on_test_start(self, trainer, pl_module):
        if self.when == 'on_test_start':
            self.exec(trainer, pl_module)

    def on_test_end(self, trainer, pl_module):
        if self.when == 'on_test_end':
            self.exec(trainer, pl_module)

    def on_predict_start(self, trainer, pl_module):
        if self.when == 'on_predict_start':
            self.exec(trainer, pl_module)

    def on_predict_end(self, trainer, pl_module):
        if self.when == 'on_predict_end':
            self.exec(trainer, pl_module)
