import os
import unittest
from pathlib import Path

import torch
from matplotlib import pyplot as plt

from src.utils.video import *
from src.utils.visualization import display_video_as_frames


class TestReadVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.video_path = Path('test_video.mp4')
        cls.start_sec = 1.1
        cls.end_sec = 3.2
        cls.frame_rate = 24
        cls.height = 720
        cls.width = 1280

        cls.total_frames = round(cls.end_sec * cls.frame_rate) - round(cls.start_sec * cls.frame_rate)
        cls.mean = 18.155
        # cls.create_test_video(cls.video_path, cls.end_sec + 1.2, cls.frame_rate, (cls.width, cls.height))

    def create_test_video(video_path: Path, duration: float, fps: int, size: tuple):
        cmd = f"""ffmpeg -f lavfi -i color=size={size[0]}x{size[1]}:rate={fps}:color=black -y -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:fontsize=360:fontcolor=white:x=(w-tw)/2:y=(h-th)/2:text='%{{frame_num}}'" -t {duration} {video_path}"""
        os.system(cmd)

    def setUp(self):
        self.reader_name = 'default'
        self.func = read_video

    def assert_video_info(self, info, height, width, frame_rate, total_frames):
        self.assertEqual(info.height, height)
        self.assertEqual(info.width, width)
        self.assertEqual(info.frame_rate, frame_rate)
        self.assertEqual(info.num_frames, total_frames)

    def assert_video(self, video, shape: tuple, mean=None, dtype=torch.uint8, delta=0.2):
        self.assertEqual(tuple(video.shape), shape)
        self.assertEqual(video.dtype, dtype)
        self.assertAlmostEqual(video.float().mean().item(), self.mean if mean is None else mean, delta=delta)

    def test_read_video(self):
        # 测试读取视频帧的基本功能
        video, info = self.func(
            video_path=self.video_path,
            start_sec=self.start_sec,
            end_sec=self.end_sec,
        )
        self.assert_video(video, (self.total_frames, self.height, self.width, 3), 18.155)
        self.assert_video_info(info, self.height, self.width, self.frame_rate, self.total_frames)

    def test_read_video_resize(self, height=360, width=640):
        video, info = self.func(
            video_path=self.video_path,
            start_sec=self.start_sec,
            end_sec=self.end_sec,
            resize=(height, width),
        )

        factor = min(self.height / height, self.width / width)
        height, width = round(self.height / factor), round(self.width / factor)
        self.assert_video(video, (self.total_frames, height, width, 3), 18.157, delta=1)
        self.assert_video_info(info, height, width, self.frame_rate, self.total_frames)

    def test_read_video_interpolation(self, height=360, width=640, interpolation='bilinear'):
        video, info = self.func(
            video_path=self.video_path,
            start_sec=self.start_sec,
            end_sec=self.end_sec,
            interpolation=interpolation,
            resize=(height, width),
        )

        factor = min(self.height / height, self.width / width)
        height, width = round(self.height / factor), round(self.width / factor)
        self.assert_video(video, (self.total_frames, height, width, 3), 18.158, delta=1)
        self.assert_video_info(info, height, width, self.frame_rate, self.total_frames)

    def test_read_video_output_format(self):
        video, info = self.func(
            video_path=self.video_path,
            start_sec=self.start_sec,
            end_sec=self.end_sec,
            output_format='TCHW',
        )
        self.assert_video(video, (self.total_frames, 3, self.height, self.width), 18.155)
        self.assert_video_info(info, self.height, self.width, self.frame_rate, self.total_frames)

    def test_read_video_num_frame(self, num_frame=16):
        video, info = self.func(
            video_path=self.video_path,
            start_sec=self.start_sec,
            end_sec=self.end_sec,
            num_frame=num_frame,
        )

        self.assert_video(video, (num_frame, self.height, self.width, 3), 18.745)
        self.assert_video_info(info, self.height, self.width, self.frame_rate, num_frame)

    def test_read_video_resize_format(self, height=360, width=640):
        video, info = self.func(
            video_path=self.video_path,
            start_sec=self.start_sec,
            end_sec=self.end_sec,
            resize=(height, width),
            output_format='TCHW',
        )

        factor = min(self.height / height, self.width / width)
        height, width = round(self.height / factor), round(self.width / factor)
        self.assert_video(video, (self.total_frames, 3, height, width), 18.157, delta=1)
        self.assert_video_info(info, height, width, self.frame_rate, self.total_frames)

    def test_read_video_real(self, height=256, width=384, num_frame=16):
        video, info = self.func(
            video_path=self.video_path,
            start_sec=self.start_sec,
            end_sec=self.end_sec,
            resize=(height, width),
            num_frame=num_frame,
            output_format='TCHW',
        )

        factor = min(self.height / height, self.width / width)
        height, width = round(self.height / factor), round(self.width / factor)
        self.assert_video(video, (num_frame, 3, height, width), 18.827)
        self.assert_video_info(info, height, width, self.frame_rate, num_frame)

    def test_read_video_real_visualization(self):
        video, info = self.func(
            video_path=self.video_path,
            start_sec=self.start_sec,
            end_sec=self.end_sec,
            resize=(256, 384),
            num_frame=16,
            output_format='TCHW',
        )
        img = display_video_as_frames(video.permute(0, 2, 3, 1), num_frames=16)
        plt.figure(figsize=(12, 1))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.title(f'{self.reader_name}')
        plt.show()


class TestReadVideoAv(TestReadVideo):
    def setUp(self):
        self.reader_name = 'av'
        self.func = read_video_av


class TestReadVideoTa(TestReadVideo):
    def setUp(self):
        self.reader_name = 'torchaudio'
        self.func = read_video_ta


class TestReadVideoTv(TestReadVideo):
    def setUp(self):
        self.reader_name = 'torchvideo'
        self.func = read_video_tv


if __name__ == '__main__':
    unittest.main()
