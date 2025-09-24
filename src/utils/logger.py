import os

from lightning.pytorch.cli import SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
from lightning_fabric.utilities.cloud_io import get_filesystem


class WandbSaveConfigCallback(SaveConfigCallback):
    def setup(self, trainer, pl_module, stage: str) -> None:
        if self.already_saved or not isinstance(trainer.logger, WandbLogger):
            return

        if self.save_to_log_dir:
            # change log dir to wandb log dir
            log_dir = str(trainer.logger.experiment.dir)
            assert log_dir is not None
            config_path = os.path.join(log_dir, self.config_filename)
            fs = get_filesystem(log_dir)

            if not self.overwrite:
                # check if the file exists on rank 0
                file_exists = fs.isfile(config_path) if trainer.is_global_zero else False
                # broadcast whether to fail to all ranks
                file_exists = trainer.strategy.broadcast(file_exists)
                if file_exists:
                    raise RuntimeError(
                        f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                        " results of a previous run. You can delete the previous config file,"
                        " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                        ' or set `LightningCLI(save_config_kwargs={"overwrite": True})` to overwrite the config file.'
                    )

            if trainer.is_global_zero:
                # save only on rank zero to avoid race conditions.
                # the `log_dir` needs to be created as we rely on the logger to do it usually
                # but it hasn't logged anything at this point
                fs.makedirs(log_dir, exist_ok=True)
                self.parser.save(
                    self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
                )

        if trainer.is_global_zero:
            self.save_config(trainer, pl_module, stage)
            self.already_saved = True

        # broadcast so that all ranks are in sync on future calls to .setup()
        self.already_saved = trainer.strategy.broadcast(self.already_saved)

    def save_config(self, trainer, pl_module, stage):
        trainer.logger.log_hyperparams(self.config.as_dict())
