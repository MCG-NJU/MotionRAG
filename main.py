import torch
from lightning.pytorch.cli import LightningCLI

from src.utils.logger import WandbSaveConfigCallback


def cli_main():
    # ignore all warnings that could be false positives
    torch.set_float32_matmul_precision('medium')
    cli = LightningCLI(auto_configure_optimizers=False,
                       save_config_callback=WandbSaveConfigCallback,
                       save_config_kwargs={"config_filename": 'CLI_config.yaml'})


if __name__ == "__main__":
    cli_main()
