from einops import rearrange
from matplotlib import pyplot as plt
from base64 import b64encode
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from IPython.display import HTML
from PIL import Image
from torchvision.io import write_video

from src.utils.video import read_video


def display_video_as_frames(video: torch.Tensor | np.ndarray | str | Path, num_frames: int = 6, start_sec: float = None,
                            end_sec: float = None, resize: Tuple[int, int] | int = 360, rows: int = 1) -> Image:
    """
    Display video as frames
    :param video: Tensor or numpy array of shape (T, H, W, C) or path to video file
    :param num_frames: Number of frames to display
    :param start_sec: Start time in seconds
    :param end_sec: End time in seconds
    :param resize: Resize video to this size
    :param rows: Number of rows
    :return: PIL Image
    """
    if isinstance(video, str) or isinstance(video, Path):
        assert Path(video).exists() and start_sec is not None and end_sec is not None
        video, _ = read_video(video_path=video, start_sec=start_sec, end_sec=end_sec, resize=resize)

    if isinstance(video, np.ndarray):
        video = torch.from_numpy(video)

    assert video.dtype == torch.uint8
    assert len(video.shape) == 4
    T, H, W, C = video.shape
    assert C == 3
    assert T >= num_frames

    video = video[np.linspace(0, T - 1, num_frames, dtype=np.int64)]

    video = rearrange(video, '(row col) h w c -> (row h) (col w) c', row=rows).numpy()

    return Image.fromarray(video)


def display_video(video: torch.Tensor | np.ndarray | str, fps: int = 8, start_sec: float = None,
                  end_sec: float = None, resize: Tuple[int, int] | int = 360, encode_speed: int = 3) -> HTML:
    """
    Display video
    :param video: Tensor or numpy array of shape (T, H, W, C) or path to video file
    :param fps: Frames per second
    :param start_sec: Start time in seconds
    :param end_sec: End time in seconds
    :param resize: Resize video to this size
    :param encode_speed: Encode speed, 0-8
    :return: HTML
    """
    if isinstance(video, str) or isinstance(video, Path):
        assert start_sec is not None and end_sec is not None
        video, info = read_video(video_path=video, start_sec=start_sec, end_sec=end_sec, resize=resize)
        fps = info.frame_rate

    if isinstance(video, np.ndarray):
        video = torch.from_numpy(video)

    assert video.dtype == torch.uint8
    assert len(video.shape) == 4
    assert video.shape[3] == 3
    assert encode_speed in range(0, 9)

    tmp_file = f'{np.random.randint(0, 10000000)}.mp4'
    write_video(tmp_file, video, fps, video_codec='libvpx-vp9', options={'crf':      '27',
                                                                         'row-mt':   '1',
                                                                         'cpu-used': f'{encode_speed}',
                                                                         'deadline': 'realtime'})

    with open(tmp_file, 'rb') as f:
        data = f.read()
    Path(tmp_file).unlink()
    html = f"""
            <video width=400 autoplay="autoplay" loop="true" controls>
                  <source src="data:video/mp4;base64,{b64encode(data).decode()}" type="video/mp4">
            </video>
            """
    return HTML(html)


def plot_motion(clip_info: dict) -> plt.Figure:
    """
    Plot motion of a clip
    :param clip_info: 
            - video: Path to video file
            - start_sec: Start time in seconds
            - end_sec: End time in seconds
            - motion: Motion histogram of the clip
    :return: Figure
    """
    ncols, nrows = 4, 3

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 6), layout='constrained', dpi=70)

    print(clip_info['video'], clip_info['start_sec'], clip_info['end_sec'])
    fig.suptitle(f'{clip_info["video"]}          {clip_info["start_sec"]:.2f}-{clip_info["end_sec"]:.2f}')

    frame_idx = np.linspace(0, len(clip_info['motion']) - 1, ncols * nrows, dtype=int)
    x = 2 ** torch.linspace(-7, 5, 256)

    for ax, i in zip(axes.flatten(), frame_idx):
        ax.plot(x, clip_info['motion'][i])
        ax.set_xscale('log')
        # ax.set_xlabel('motion magnitude')
        # ax.set_ylabel('count')
        # hide ticks
        ax.set_xticks([])
        ax.set_yticks([])

    return fig
