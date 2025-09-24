import argparse
from typing import Any

from lightning import seed_everything
from lightning.pytorch.callbacks import RichProgressBar
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import Transform, Compose

from .callbacks import *
from ..utils.video import read_video


class VideoReader(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generated_videos: list[dict] = []

    def validation_step(self, batch, batch_idx):
        video = batch["video"]
        gt_video = batch["gt_video"]
        idx = batch["video_id"]

        self.generated_videos.append({
            "video":    video,
            "gt_video": gt_video,
            "video_id": idx,
        })

    def test_step(self, *args: Any, **kwargs: Any):
        return self.validation_step(*args, **kwargs)

    def on_validation_start(self) -> None:
        self.generated_videos.clear()

    def on_test_start(self) -> None:
        self.on_validation_start()


class VideoDataset(Dataset):
    def __init__(self,
                 gt_video_dir: str,
                 generated_video_dir: str,
                 transforms: Transform = None,
                 annotations: list[dict] = None,
                 ):
        """
        Args:
            gt_video_dir: path to the ground truth video directory
            generated_video_dir: path to the generated video directory
            transforms: transforms applied to the video
            annotations: annotations for the dataset, optional
        """
        self.transforms = transforms
        self.gt_video_dir = Path(gt_video_dir)
        self.generated_video_dir = Path(generated_video_dir)
        self.annotations = annotations

        self.gt_video = sorted(self.gt_video_dir.glob("*.mp4"), key=lambda x: int(x.stem))
        self.generated_video = sorted(self.generated_video_dir.glob("*.mp4"), key=lambda x: int(x.stem))
        assert len(self.gt_video) == len(
            self.generated_video), f"Num of videos in gt_video_dir and generated_video_dir must be the same, but got {len(self.gt_video)} and {len(self.generated_video)}"

    def __len__(self):
        return len(self.gt_video)

    def __getitem__(self, idx) -> dict:
        try:
            gt_video, _ = read_video(self.gt_video[idx], start_sec=0, end_sec=2, threads=2, output_format='TCHW')
            generate_video, _ = read_video(self.generated_video[idx], start_sec=0, end_sec=2, threads=2,
                                           output_format='TCHW')

            if self.transforms is not None:
                gt_video = self.transforms(gt_video)
                generate_video = self.transforms(generate_video)
        except Exception as e:
            print(e)

        return {'gt_video': gt_video, 'video': generate_video, 'video_id': torch.tensor(idx)}


def calc_metrics(gt_path: str, generate_path: str, num_gpu: int = 8, annotations: list[dict] = None) -> None:
    """
    Calculate metrics for a given video path
    :param gt_path: gt video path
    :param generate_path: generate video path
    :param num_gpu: number of gpu
    :param annotations: annotations for the dataset
    :return:
    """
    seed_everything(7)
    module = VideoReader()

    dataloader = DataLoader(
        VideoDataset(gt_video_dir=gt_path, generated_video_dir=generate_path, transforms=None, annotations=annotations),
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(
        accelerator="gpu",
        strategy='auto',
        devices=num_gpu,
        precision='16-mixed',
        callbacks=[
            RichProgressBar(),
            FVDMetric(),
            FIDMetric(),
            MotionMetric(),
            ActionMetric(),
            MAEActionMetric(),
            ClipV2VMetric(),
            SaveSampleMetrics(save_path=Path(gt_path).parent, create_subdir=False),
        ],
        benchmark=True,
    )

    trainer.test(module, dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=str, default="")
    parser.add_argument("--generate_path", type=str, default="")
    parser.add_argument("--num_gpu", type=int, default=8)
    parser.add_argument("--annotation_path", type=str, default=None)
    args = parser.parse_args()

    assert Path(args.gt_path).exists(), "gt_path does not exist"
    assert Path(args.generate_path).exists(), "generate_path does not exist"

    calc_metrics(args.gt_path, args.generate_path, args.num_gpu,
                 torch.load(args.annotation_path) if args.annotation_path else None)
