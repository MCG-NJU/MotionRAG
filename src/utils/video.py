import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Literal

import av
import torch
import numpy as np
from einops import rearrange
from torch import Tensor
from torchaudio.io import StreamReader
import torchvision.transforms.v2
import torchvision.io as tvio
from torchvision.transforms import InterpolationMode


@dataclass
class VideoInfo:
    height: int
    width: int
    frame_rate: float
    num_frames: int


def get_video_reader(
        video_path: str | Path,
        start_sec: float,
        end_sec: float,
        resize: Tuple[int, int] | int = None,
        interpolation: str = 'bicubic',
        threads: int = 20,
        frames_pre_chunk: int = None,
        buffer_chunk_size: int = 1,
        check_len: bool = False
) -> tuple[StreamReader, VideoInfo]:
    """
    Get StreamReader object for video
    :param video_path: Path to video file
    :param start_sec: Start time in seconds
    :param end_sec: End time in seconds
    :param resize: Resize video to this size (H, W)
    :param interpolation: Interpolation mode, available: "fast_bilinear", "bilinear", "bicubic", "neighbor", "area", "bicublin", "gauss", "sinc", "lanczos", "spline", "print_info", "accurate_rnd", "full_chroma_int
    :param threads: Number of threads to use
    :param frames_pre_chunk: Number of frames to read per chunk
    :param buffer_chunk_size: number of chunks to buffer
    :param check_len: Check if end frame is less than number of frames in video
    :return: StreamReader object and Video info
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f'{video_path}   File not found')

    streamer = StreamReader(video_path)
    for i in range(streamer.num_src_streams):
        info = streamer.get_src_stream_info(i)
        if info.media_type == 'video':
            break
    else:
        raise ValueError(f'{video_path}   No video stream found')

    if isinstance(resize, int):
        resize = (resize, resize)
    elif resize is None:
        resize = (info.height, info.width)
    factor = min(info.height / resize[0], info.width / resize[1])
    h, w = round(info.height / factor), round(info.width / factor)

    fps = info.frame_rate
    start_idx, end_idx = round(start_sec * fps), round(end_sec * fps)
    if check_len and (end_idx > info.num_frames or start_idx < 0 or end_idx <= start_idx):
        raise ValueError(f'{video_path}   End frame {end_idx} is exceeds the number of frames {info.num_frames}')

    frames_pre_chunk = end_idx - start_idx if frames_pre_chunk is None else frames_pre_chunk

    streamer.add_video_stream(frames_pre_chunk, buffer_chunk_size,
                              decoder_option={"threads": f"{threads}"},
                              filter_desc=f'scale={w}:{h}:sws_flags={interpolation},format=pix_fmts=rgb24')
    streamer.seek(start_sec - 1 / fps, mode='precise')

    return streamer, VideoInfo(h, w, fps, info.num_frames)


def read_video_ta(
        video_path: str | Path,
        start_sec: float,
        end_sec: float,
        resize: Tuple[int, int] | int = None,
        interpolation: str = 'bicubic',
        threads: int = 20,
        output_format: Literal['THWC', 'TCHW'] = 'THWC',
        num_frame: int = None,
) -> tuple[Tensor, VideoInfo]:
    """
    Read video from file using torchaudio
    :param video_path: Path to video file
    :param start_sec: Start time in seconds
    :param end_sec: End time in seconds
    :param resize: Resize video to this size (H, W)
    :param interpolation: Interpolation mode, available: "fast_bilinear", "bilinear", "bicubic", "neighbor", "area", "bicublin", "gauss", "sinc", "lanczos", "spline", "print_info", "accurate_rnd", "full_chroma_int", "full_chroma_inp", "bitexact"
    :param threads: Number of threads to use
    :param output_format: Format of the output tensor, available: "THWC", "TCHW"
    :param num_frame: Number of frames to sample uniformly, if None, read all frames
    :return: Tensor of shape (T, H, W, C) or (T, C, H, W) and video info
    """
    streamer, info = get_video_reader(video_path, start_sec, end_sec, resize, interpolation, threads,
                                      frames_pre_chunk=10, buffer_chunk_size=3)
    total_frame = max(round(end_sec * info.frame_rate) - round(start_sec * info.frame_rate), 1) + 1

    video = []
    try:
        for (chunk,) in streamer.stream():
            if total_frame <= 0:
                break
            video.append(chunk[:total_frame])
            total_frame -= len(chunk[:total_frame])
    except Exception as e:
        raise ValueError(f'{video_path}   Read video failed: {e}')

    # BUGS: ignore the case where the last frame is not read
    if total_frame > 1:
        raise ValueError(f'{video_path}   Read video failed: {total_frame} frames missing')

    video = torch.cat(video, dim=0)

    fps = info.frame_rate
    num_frame = len(video) - 1 if num_frame is None else num_frame
    frame_idx = np.linspace(start_sec * fps, end_sec * fps - 1, num_frame).round() - math.floor(start_sec * fps)
    video = video[frame_idx]

    if output_format == 'THWC':
        video = rearrange(video, 't c h w -> t h w c')

    info.num_frames = len(video)

    return video, info


def read_video_av(
        video_path: str | Path,
        start_sec: float,
        end_sec: float,
        resize: Tuple[int, int] | int = None,
        interpolation: str = 'bicubic',
        threads: int = 1,
        output_format: Literal['THWC', 'TCHW'] = 'THWC',
        num_frame: int = None,
) -> tuple[Tensor, VideoInfo]:
    """
    Read video from file using av
    :param video_path: Path to video file
    :param start_sec: Start time in seconds
    :param end_sec: End time in seconds
    :param resize: Resize video to this size (H, W)
    :param interpolation: Interpolation mode, available: "fast_bilinear", "bilinear", "bicubic", "neighbor", "area", "bicublin", "gauss", "sinc", "lanczos", "spline", "print_info", "accurate_rnd", "full_chroma_int", "full_chroma_inp", "bitexact"
    :param threads: Number of threads to use
    :param output_format: Format of the output tensor, available: "THWC", "TCHW"
    :param num_frame: Number of frames to sample uniformly, if None, read all frames
    :return: Tensor of shape (T, H, W, C) or (T, C, H, W) and video info
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f'{video_path}   File not found')

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        stream.thread_count = threads

        if isinstance(resize, int):
            resize = (resize, resize)
        elif resize is None:
            resize = (stream.height, stream.width)
        factor = min(stream.height / resize[0], stream.width / resize[1])
        h, w = round(stream.height / factor), round(stream.width / factor)

        fps = stream.average_rate

        if num_frame is None:
            num_frame = max(round(end_sec * fps) - round(start_sec * fps), 1)

        info = VideoInfo(h, w, float(fps), num_frame)

        start_pts, end_pts = round(start_sec / stream.time_base), round(end_sec / stream.time_base)
        delta = 1 / stream.average_rate / stream.time_base // 2
        timestamps = np.linspace(start_pts, end_pts - delta * 2, num_frame, dtype=int).tolist()

        video = np.empty((num_frame, h, w, 3), dtype=np.uint8)
        try:
            idx, pts = 0, timestamps.pop(0)
            if start_pts > 0:
                container.seek(start_pts, stream=stream)
            for frame in container.decode(stream):
                if frame.pts >= pts - delta:
                    frame = frame.reformat(width=w, height=h, interpolation=interpolation.upper())
                    video[idx] = frame.to_ndarray(format='rgb24')  # convert to RGB after resize is faster

                    if len(timestamps) == 0:
                        break
                    else:
                        idx, pts = idx + 1, timestamps.pop(0)

            if len(timestamps) != 0:
                raise ValueError(f'{video_path}   Read video failed: {len(timestamps)} frames missing')

        except Exception as e:
            raise ValueError(f'{video_path}   Read video failed: {e}')
        finally:
            stream.close()

    if output_format == 'TCHW':
        video = video.transpose(0, 3, 1, 2)  # T H W C -> T C H W

    return torch.from_numpy(video).contiguous(), info


def read_video_tv(
        video_path: str | Path,
        start_sec: float,
        end_sec: float,
        resize: Tuple[int, int] | int = None,
        interpolation: str = 'bicubic',
        threads: int = 1,
        output_format: Literal['THWC', 'TCHW'] = 'THWC',
        num_frame: int = None,
) -> tuple[Tensor, VideoInfo]:
    """
    Read video from file using torchvision
    :param video_path: Path to video file
    :param start_sec: Start time in seconds
    :param end_sec: End time in seconds
    :param resize: Resize video to this size (H, W)
    :param interpolation: Interpolation mode, available: "bilinear", "bicubic", "nearest", "lanczos", "box", "hamming"
    :param threads: Number of threads to use
    :param output_format: Format of the output tensor, available: "THWC", "TCHW"
    :param num_frame: Number of frames to sample uniformly, if None, read all frames
    :return: Tensor of shape (T, H, W, C) or (T, C, H, W) and video info
    """
    video, audio, info = tvio.read_video(video_path,
                                         start_pts=start_sec,
                                         end_pts=end_sec,
                                         pts_unit='sec',
                                         output_format="TCHW")
    fps = info['video_fps']

    total_frame = round(end_sec * info['video_fps']) - round(start_sec * info['video_fps'])
    num_frame = total_frame if num_frame is None else num_frame
    frame_idx = np.linspace(start_sec * fps, end_sec * fps - 1, num_frame).round() - math.floor(start_sec * fps)

    video = video[frame_idx]

    height, width = video.shape[2:]
    if resize is not None:
        if isinstance(resize, int):
            resize = (resize, resize)

        factor = min(height / resize[0], width / resize[1])
        height, width = round(height / factor), round(width / factor)

        video = torchvision.transforms.v2.Resize(size=(height, width),
                                                 interpolation={'bicubic':  InterpolationMode.BICUBIC,
                                                                'bilinear': InterpolationMode.BILINEAR,
                                                                'nearest':  InterpolationMode.NEAREST,
                                                                'box':      InterpolationMode.BOX,
                                                                'hamming':  InterpolationMode.HAMMING,
                                                                'lanczos':  InterpolationMode.LANCZOS, }[interpolation],
                                                 antialias=True)(video)

    info = VideoInfo(height, width, fps, len(video))

    if output_format == 'THWC':
        video = video.permute(0, 2, 3, 1)

    return video, info


read_video = read_video_av
