from pathlib import Path
from typing import Any

import lightning.pytorch as pl
from PIL import Image
from lightning.pytorch.utilities.types import STEP_OUTPUT


class SaveLastFrame(pl.Callback):
    def __init__(self, save_path: str | Path = 'ref_frame/ar'):
        self.save_path = Path(save_path)

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        # [b f c h w]
        for info, video in zip([d['annotation'] for d in batch['metadata']], outputs):
            # [c h w]->[h w c]
            last_frame = video[-1].permute(1, 2, 0).numpy()
            if 'clip_id' in info:
                Image.fromarray(last_frame).save(self.save_path / f"{info['clip_id']}.png")
            else:
                trainer.print('No clip_id found in metadata')

    def on_test_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        return self.on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
