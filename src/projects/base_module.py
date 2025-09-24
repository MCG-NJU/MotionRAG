import gc
from typing import Callable, Iterable, Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ..utils.lora_utils import insert_lora_module
from ..utils.pipeline import denormalize

OptimizerCallable = Callable[[Iterable], Optimizer]
LRSchedulerCallable = Callable[[Optimizer], LRScheduler]


class BaseModule(pl.LightningModule):
    def __init__(self,
                 optimizer: OptimizerCallable = None,
                 lr_scheduler: LRSchedulerCallable = None,
                 freeze_modules: Iterable[str] = None,
                 full_trainable_modules: Iterable[str] = None,
                 lora_trainable_modules: Iterable[str] = None,
                 lora_rank: int = 64,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.freeze_modules = freeze_modules if freeze_modules is not None else []
        self.full_trainable_modules = full_trainable_modules if full_trainable_modules is not None else []
        self.lora_trainable_modules = lora_trainable_modules if lora_trainable_modules is not None else []
        self.lora_rank = lora_rank
        self._model_configured = False
        self.trainable_parameters = None
        self.strict_loading = False

        if type(self).configure_model != BaseModule.configure_model:
            raise RuntimeError(
                "You should not override configure_model in your module! Please override configure_model_ instead.")

    def get_trainable_parameters(self):
        for name, module in self.named_modules():
            if name in self.freeze_modules:
                module.requires_grad_(False)
                module.eval()
                # print(f"Freezing {name}")

        def _get_full_parameters():
            for name, module in self.named_modules():
                if name in self.full_trainable_modules:
                    module.requires_grad_(True)
                    module.train()
                    yield from module.parameters()

        def _get_lora_parameters():
            for name in self.lora_trainable_modules:
                lora_layer = insert_lora_module(self, name, rank=self.lora_rank)
                assert lora_layer is not None
                yield from lora_layer.parameters()
            gc.collect()

        trainable_parameters = list(_get_lora_parameters()) + list(_get_full_parameters())
        return trainable_parameters

    def configure_model(self) -> None:
        """You should not override this method in your module. Please override 'configure_model_' instead."""
        if not self._model_configured:
            self.configure_model_()

            # trainable parameters must be set inside configure_model, otherwise DDP will detect unused parameters.
            self.trainable_parameters = self.get_trainable_parameters()

            self._model_configured = True

    def configure_model_(self) -> None:
        """
        Override this method to configure your model.
        """
        raise NotImplementedError

    def configure_optimizers(self):
        if self.optimizer is None:
            return
        self.optimizer = self.optimizer(self.trainable_parameters)

        if self.lr_scheduler is None:
            return self.optimizer

        self.lr_scheduler = self.lr_scheduler(self.optimizer)  # if self.scheduler else None
        return dict(
            optimizer=self.optimizer,
            lr_scheduler={
                "scheduler": self.lr_scheduler,
                "interval":  "step",
            })


class VideoBaseModule(BaseModule):
    def __init__(self,
                 optimizer: OptimizerCallable = None,
                 lr_scheduler: LRSchedulerCallable = None,
                 freeze_modules: Iterable[str] = None,
                 full_trainable_modules: Iterable[str] = None,
                 lora_trainable_modules: Iterable[str] = None,
                 lora_rank: int = 64,
                 eval_pipeline_call_kwargs: dict = None,
                 *args,
                 **kwargs
                 ):
        super().__init__(optimizer, lr_scheduler, freeze_modules, full_trainable_modules, lora_trainable_modules,
                         lora_rank, *args, **kwargs)
        self.eval_pipeline_call_kwargs = {}
        for k, v in eval_pipeline_call_kwargs.items():
            if isinstance(v, str):
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        v = v
            self.eval_pipeline_call_kwargs[k] = v

        self.generated_videos: list[dict] = []

    def validation_step(self, batch, batch_idx) -> Tensor:
        metadata = batch["metadata"]
        image_condition = batch['ref_frame']  # image_condition: [B C H W]
        positive_prompt = [b['raw_prompt'] for b in metadata]
        negative_prompt = [''] * len(positive_prompt)
        ref_videos = batch['ref_videos']  # [b k f c h w]

        generate_videos = self.eval_pipeline(
            image=image_condition,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            dtype=self.dtype,
            ref_videos=ref_videos,
            metadata=metadata,
            **self.eval_pipeline_call_kwargs)

        return denormalize(generate_videos).cpu()  # generate_videos: [b f c h w]

    def test_step(self, *args: Any, **kwargs: Any):
        return self.validation_step(*args, **kwargs)

    @staticmethod
    def output_assertions(outputs: Tensor, batch: Any) -> None:
        assert isinstance(outputs, Tensor), f"Expected outputs to be a tensor, got {type(outputs)}"
        assert outputs.dtype == torch.uint8, f"Expected outputs to be uint8, got {outputs.dtype}"
        assert outputs.device == torch.device("cpu"), f"Expected outputs to be on CPU, got {outputs.device}"
        assert len(outputs.shape) == 5, f"Expected outputs to be 5D, got {len(outputs.shape)}D"
        assert 'metadata' in batch, f"Expected batch to have a 'metadata' attribute, got {batch}"
        assert len(batch['metadata']) == outputs.size(0), \
            f"Metadata length does not match outputs batch size, got {len(batch['metadata'])}, expected {outputs.size(0)}"

    def on_validation_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int,
                                dataloader_idx: int = 0) -> None:
        self.output_assertions(outputs, batch)

        metadata: list[dict] = batch['metadata']
        gt_videos = denormalize(batch['video']).cpu() if 'video' in batch else None

        for i, metadata_item in enumerate(metadata):
            self.generated_videos.append({
                'video':     outputs[i][None],  # [1 f c h w]
                'gt_video':  gt_videos[i][None] if gt_videos is not None else None,  # [1 f c h w]
                'id':        metadata_item['id'],
                'prompt':    metadata_item['raw_prompt'],
                'save_name': metadata_item['save_name'],
            })

    def on_test_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        return self.on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def on_validation_start(self) -> None:
        self.generated_videos.clear()

    def on_test_start(self) -> None:
        self.generated_videos.clear()


class ImageBaseModule(VideoBaseModule):
    def validation_step(self, batch, batch_idx) -> list[Tensor]:
        raise NotImplementedError

    @staticmethod
    def output_assertions(outputs: Tensor, batch: Any) -> None:
        assert isinstance(outputs, list), f"Expected outputs to be a list, got {type(outputs)}"
        for output in outputs:
            assert isinstance(output, Tensor), f"Expected outputs to be a tensor, got {type(output)}"
            assert output.dtype == torch.uint8, f"Expected outputs to be uint8, got {output.dtype}"
            assert output.device == torch.device("cpu"), f"Expected outputs to be on CPU, got {output.device}"
            assert len(output.shape) == 5, f"Expected outputs to be 5D, got {len(output.shape)}D"
        assert 'metadata' in batch, f"Expected batch to have a 'image' attribute, got {batch}"
        assert isinstance(batch['metadata'], list), \
            f"Expected batch['metadata'] to be a list, got {type(batch['metadata'])}"
        assert len(outputs) == sum(len(m) - 1 for m in batch['metadata']), \
            f"Metadata length does not match outputs batch size, expected {outputs.size(0)}"

    def on_validation_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int,
                                dataloader_idx: int = 0) -> None:
        self.output_assertions(outputs, batch)
        metadata: list[list[dict]] = batch['metadata']

        cnt = 0
        for metadata_item in metadata:
            for m in metadata_item[1:]:  # ignore first image
                self.generated_videos.append({
                    'video':     outputs[cnt],  # [1 f c h w]
                    'gt_video':  m['original_image'][None],  # [1 f c h w]
                    'id':        m['id'],
                    'prompt':    m['raw_prompt'],
                    'save_name': m['save_name'],
                })
                cnt += 1
