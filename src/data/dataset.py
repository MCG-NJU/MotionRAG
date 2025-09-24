import random
import time
from pathlib import Path
from typing import Tuple, Literal

import PIL.Image
import numpy as np
import torch
import torchvision.transforms.v2 as tvtf
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from transformers.models.clip import CLIPTokenizer

from src.utils.video import read_video_av, read_video_ta, read_video_tv


def collate_fn(batch):
    """
    :param batch: list[dict]
    :return: dict
        video: torch.tensor [B T C H W]
        prompt: torch.tensor [B 77] or list[str]
        ref_frame: torch.tensor [B C H W]
        ref_videos: torch.tensor [B K T C H W]
        metadata: list[dict]
    """
    video = torch.cat([b['video'] for b in batch], dim=0)
    prompt = [b['prompt'] for b in batch]
    prompt = torch.cat(prompt, dim=0) if isinstance(prompt[0], Tensor) else prompt
    ref_frame = torch.cat([b['ref_frame'] for b in batch], dim=0)
    ref_videos = torch.stack([b['ref_videos'] for b in batch], dim=0)
    metadata = [b['metadata'] for b in batch]
    return {
        "video":      video,
        "prompt":     prompt,
        "ref_frame":  ref_frame,
        "ref_videos": ref_videos,
        "metadata":   metadata
    }


class VideoDataset(Dataset):
    def __init__(self,
                 annotation_path: Path | str,
                 video_size: Tuple[int, int],
                 video_length: int,
                 video_dir: str,
                 extra_transforms: Tuple = tuple(),
                 read_video_fn: Literal['av', 'ta', 'tv'] = 'av',
                 tokenizer_model: str | None = None,
                 sampling_config: dict[int, float] = None,
                 uncond_text_ratio: float = 0.15,
                 uncond_video_ratio: float = 0.15,
                 use_ref_frame: bool = False,
                 ref_frame_dir: str | Path = None,
                 prompt_type: Literal["llm", "image", "video", "action", "llm_plan", 'mix'] = 'llm',
                 ref_video_num: int = 1,
                 ):
        """
        Args:
            annotation_path: Path
            video_size: (Height, Width)
            video_length: length of the video in frames
            video_dir: path to the video directory
            extra_transforms: extra transforms to apply to the video
            read_video_fn: read video function, can be 'av'(pyav) 'tv'(torchvision) or 'ta'(torchaudio)
            tokenizer_model: path to the tokenizer model
            sampling_config: video sampling rate (fps) and its probability, i.e. {4: 0.2, 8: 0.5, 12: 0.3}
            uncond_text_ratio: ratio of unconditional text
            use_ref_frame: use reference frame
            prompt_type: prompt type
            ref_video_num: number of reference videos
        """
        self.annotations = torch.load(annotation_path)
        self.video_dir = Path(video_dir)
        if tokenizer_model is not None:
            try:
                self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_model, subfolder="tokenizer")
            except:
                self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_model)
        else:
            self.tokenizer = None

        self.transforms = tvtf.Compose([
            tvtf.CenterCrop(video_size),
            *extra_transforms,
            tvtf.ToDtype(torch.bfloat16, scale=True),  # 0-1
            tvtf.Normalize(mean=[0.5, ] * 3, std=[0.5, ] * 3, inplace=True),  # -1 1
        ])

        self.ref_image_transforms = tvtf.Compose([
            tvtf.CenterCrop(video_size),
            *extra_transforms,
            tvtf.ToDtype(torch.bfloat16, scale=True),  # 0-1
            tvtf.Normalize(mean=[0.5, ] * 3, std=[0.5, ] * 3, inplace=True),  # -1 1
        ])

        # video length (frames)
        self.video_length = video_length
        self.video_size = video_size

        # sampling config
        self.sampling_config = {8: 1} if sampling_config is None else sampling_config

        self.uncond_text_ratio = uncond_text_ratio
        self.uncond_video_ratio = uncond_video_ratio
        self.prompt_type = prompt_type
        self.use_ref_frame = use_ref_frame
        self.ref_frame_dir = Path(ref_frame_dir) if ref_frame_dir is not None else Path('.')
        self.ref_video_num = ref_video_num

        self.read_video_fn = {'av': read_video_av, 'tv': read_video_tv, 'ta': read_video_ta}[read_video_fn]

    def __len__(self):
        return len(self.annotations)

    def video_clip_sampler(self, start_sec: float, end_sec: float, sampling_config: dict[int, float] = None) -> Tuple[
        float, float]:
        """
        Sample a subclip from the large video clip uniformly.
        :param start_sec: start second of the large clip
        :param end_sec: end second of the large clip
        :param sampling_config: dict of sampling rate and its probability, i.e. {4: 0.2, 8: 0.5, 12: 0.3}
        :return: start_sec, end_sec of the subclip
        """
        sampling_config = self.sampling_config if sampling_config is None else sampling_config
        clip_max_length = self.video_length / np.random.choice(list(sampling_config.keys()),
                                                               p=list(sampling_config.values()))

        if end_sec - start_sec > clip_max_length:
            start_sec = random.uniform(start_sec, end_sec - clip_max_length)
            end_sec = start_sec + clip_max_length
        return start_sec, end_sec

    def __getitem__(self, idx: int) -> dict:
        try:
            return self.getitem(idx)
        except Exception as e:
            print(f"Data Error: {e}   Batch idx:{idx}")
            return self.__getitem__(random.randint(0, len(self) - 1))

    def getitem(self, idx: int) -> dict:
        """
        Get a video clip and its prompt.
        :param idx:
        :return: dict
            video: torch.tensor [1 T C H W]
            prompt: torch.tensor [1 77]
            ref_frame: torch.tensor [1 C H W]
            ref_videos: torch.tensor [K T C H W]
            metadata: dict
        """
        video_info = self.annotations[idx]

        video_data = self.get_video(video_info)
        video = video_data['video']

        prompt, raw_prompt = self.get_prompt(video_info)

        ref_frame = self.get_ref_frame(video, video_info)

        ref_videos, ref_video_distance = self.get_ref_videos(video, video_info)

        metadata = {
            "raw_prompt":         raw_prompt,
            "info":               video_data['info'],
            "read_video_time":    video_data['read_video_time'],
            "transforms_time":    video_data['transforms_time'],
            "clip_length":        video_data['end_sec'] - video_data['start_sec'],
            "batch_idx":          idx,
            "id":                 video_info['id'],
            "save_name":          video_info['save_name'] if 'save_name' in video_info else video_info['id'],
            "ref_video_distance": ref_video_distance,
            "annotation":         {k: v for k, v in video_info.items() if not isinstance(v, torch.Tensor)}
            # ignore tensor
        }
        return {
            "video":      video,
            "prompt":     prompt,
            "ref_frame":  ref_frame,
            "ref_videos": ref_videos,
            "metadata":   metadata
        }

    def get_video(self, video_info: dict, sampling_config: dict[int, float] = None) -> dict:
        """
        Get a video clip from file.
        :param video_info: dict
        :param sampling_config: dict of sampling rate and its probability, i.e. {4: 0.2, 8: 0.5, 12: 0.3}
        :return: dict
            video: torch.tensor [1 T C H W]
            start_sec: start second of the video clip
            end_sec: end second of the video clip
            info: dict of video info
            read_video_time: time spent on reading video
            transforms_time: time spent on transforms
        """
        # sample a subclip
        start_sec, end_sec = self.video_clip_sampler(video_info['start_sec'], video_info['end_sec'], sampling_config)
        # read video
        start_read_video = time.time()

        num_frame = 1 if start_sec == end_sec else self.video_length  # case of single frame
        video, info = self.read_video_fn(self.video_dir / video_info['video'], start_sec=start_sec, end_sec=end_sec,
                                         resize=self.video_size, interpolation="bicubic", output_format="TCHW",
                                         num_frame=num_frame)

        read_video_time = time.time() - start_read_video
        # transform
        start_transform = time.time()
        video = self.transforms(video[None, ...])
        transforms_time = time.time() - start_transform

        return {
            "video":           video,  # [1 T C H W]
            "start_sec":       start_sec,
            "end_sec":         end_sec,
            "info":            info,
            "read_video_time": read_video_time,
            "transforms_time": transforms_time,
        }

    def get_prompt(self, video_info: dict) -> tuple[Tensor, str]:
        """
        Get prompt(token ids) and raw prompt(str)
        :param video_info: dict
        :return: Tuple
            prompt: torch.tensor [1 77]
            raw_prompt: str
        """
        if self.prompt_type == 'llm':
            raw_prompt = video_info['llm_caption']
        elif self.prompt_type == 'image':
            raw_prompt = random.choice(video_info['image_caption'])
        elif self.prompt_type == 'video':
            raw_prompt = random.choice(video_info['video_caption'])
        elif self.prompt_type == 'action':
            raw_prompt = video_info['prompt']
        elif self.prompt_type == 'llm_plan':
            raw_prompt = video_info['step_descriptions']
        elif self.prompt_type == 'mix':
            raw_prompt = random.choice([video_info['llm_caption'], random.choice(video_info['image_caption'])])
        else:
            raise ValueError("Invalid prompt type.")

        raw_prompt = raw_prompt if raw_prompt is not None else ''
        prompt = raw_prompt if random.random() > self.uncond_text_ratio else ''

        if self.tokenizer is not None:
            prompt = self.tokenizer(prompt,
                                    return_tensors="pt",
                                    padding="max_length",
                                    truncation=True, ).input_ids
        return prompt, raw_prompt

    def get_ref_frame(self, video: Tensor, video_info: dict) -> Tensor:
        """
        Get reference frame from video or image
        :param video: torch.tensor [1 T C H W]
        :param video_info: dict
        :return: torch.tensor [1 C H W]
        """
        if self.use_ref_frame and 'ref_frame' in video_info:
            ref_file = self.ref_frame_dir / video_info['ref_frame']
            if not ref_file.exists():
                print(f"Ref frame {ref_file} not found, waiting...")
                while not ref_file.exists():
                    time.sleep(1)
            # wait for 0.3 seconds to make sure the ref frame is saved correctly
            time.sleep(0.3)
            image = PIL.Image.open(ref_file).convert("RGB")

            factor = min(image.height / self.video_size[0], image.width / self.video_size[1])
            h, w = round(image.height / factor), round(image.width / factor)

            image = image.resize((w, h), resample=PIL.Image.BICUBIC)
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None, None]
            ref_frame = self.ref_image_transforms(image)[:, 0]
        else:
            # ref_frame: [1 C H W]
            ref_frame = video[:, 0]
        return ref_frame

    def get_ref_videos(self, video: Tensor, video_info: dict) -> tuple[Tensor, list[float]]:
        """
        Get reference videos.
        :param video: torch.tensor [1 T C H W]
        :param video_info: dict
        :return: torch.tensor [K(ref_video_num) T C H W], list[float]
        """
        ref_videos = torch.zeros(self.ref_video_num, self.video_length, *video.shape[2:], dtype=video.dtype,
                                 device=video.device)
        distance = []
        if 'ref_videos' in video_info:
            for i, v in enumerate(video_info['ref_videos'][:self.ref_video_num]):
                if random.random() > self.uncond_video_ratio:
                    try:
                        if v['video'] == video_info['video']:
                            ref_video = video
                        else:
                            ref_video = self.get_video(v, {8: 1})['video']

                        ref_videos[i] = ref_video
                        distance.append(v['_distance'])
                    except Exception as e:
                        print(f"Rag read video Error: {e}")
                        distance.append(1.0)
                else:
                    distance.append(1.0)

        return ref_videos, distance


class SkillImageDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.resize_transforms = tvtf.Compose([
            tvtf.Resize(max(self.video_size) - 1, interpolation=InterpolationMode.BICUBIC,
                        max_size=max(self.video_size)),
        ])

        self.transforms = tvtf.Compose([
            tvtf.CenterCrop(self.video_size),
            tvtf.ToDtype(torch.float32, scale=True),  # 0-1
            tvtf.Normalize(mean=[0.5, ] * 3, std=[0.5, ] * 3, inplace=True),  # -1 1
        ])
        self.prompt_cache = None
        self.extra_prompts = ['']

    def get_video(self, video_info: dict, sampling_config: dict[int, float] = None) -> dict:
        """
        Get a video clip from file.
        :param video_info: dict
        :param sampling_config: dict of sampling rate and its probability, i.e. {4: 0.2, 8: 0.5, 12: 0.3}
        :return: dict
            video: torch.tensor [1 T C H W]
            start_sec: start second of the video clip
            end_sec: end second of the video clip
            info: dict of video info
            read_video_time: time spent on reading video
            transforms_time: time spent on transforms
        """
        # sample a subclip
        start_sec, end_sec = self.video_clip_sampler(video_info['start_sec'], video_info['end_sec'], sampling_config)
        # read video
        start_read_video = time.time()
        video, info = self.read_video_fn(self.video_dir / video_info['video'], start_sec=start_sec, end_sec=end_sec,
                                         output_format="TCHW", num_frame=self.video_length)
        read_video_time = time.time() - start_read_video

        original_size = video.shape[2:]
        target_size = self.video_size
        orig_video = video[None]

        # transform
        start_transform = time.time()

        video = self.resize_transforms(video[None, ...])
        resized_size = video.shape[3:]
        # crop_coords_top_left = ((resized_size[0] - target_size[0]) // 2, (resized_size[1] - target_size[1]) // 2)
        crop_coords_top_left = (0, 0)
        video = self.transforms(video)

        transforms_time = time.time() - start_transform

        return {
            "video":                video,  # [1 T C H W]
            "original_video":       orig_video,
            "start_sec":            start_sec,
            "end_sec":              end_sec,
            "info":                 info,
            "original_size":        original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size":          target_size,
            "resized_size":         resized_size,
            "read_video_time":      read_video_time,
            "transforms_time":      transforms_time,
        }

    def getitem(self, idx: int) -> dict:
        """
        Get a video clip and its prompt.
        :param idx:
        :return: dict
            images: torch.tensor [N C H W]
            prompts: list[str]
            metadata: list[dict]
        """
        skill_info = self.annotations[idx]
        step_data = []

        for step_info in skill_info['steps']:
            step_info['end_sec'] = step_info['start_sec']
            video_data = self.get_video(step_info)
            image = video_data['video'][0]

            _, raw_prompt = self.get_prompt(step_info)
            prompt = raw_prompt if random.random() > self.uncond_text_ratio else ''

            metadata = {
                "raw_prompt":           raw_prompt,
                "info":                 video_data['info'],
                "read_video_time":      video_data['read_video_time'],
                "transforms_time":      video_data['transforms_time'],
                "clip_length":          video_data['end_sec'] - video_data['start_sec'],
                "id":                   step_info['id'],
                "save_name":            step_info['save_name'] if 'save_name' in step_info else step_info['id'],
                "annotation":           {k: v for k, v in step_info.items() if not isinstance(v, torch.Tensor)},
                # ignore tensor
                "original_size":        video_data['original_size'],
                "target_size":          video_data['target_size'],
                "crop_coords_top_left": video_data['crop_coords_top_left'],
                "resized_size":         video_data['resized_size'],
                "original_image":       video_data['original_video'][0],
            }

            step_data.append({
                "image":    image,
                "prompt":   prompt,
                "metadata": metadata
            })

        images = torch.concat([d['image'] for d in step_data])
        prompts = [d['prompt'] for d in step_data]
        metadata = [d['metadata'] for d in step_data]

        return {
            "images":   images,
            "prompts":  prompts,
            "metadata": metadata,
        }

    def get_all_prompts(self) -> list[str]:
        annotations = [step for skill in self.annotations for step in skill['steps']]
        if self.prompt_type == 'llm':
            prompts = [[anno['llm_caption']] for anno in annotations]
        elif self.prompt_type == 'image':
            prompts = [anno['image_caption'] for anno in annotations]
        elif self.prompt_type == 'video':
            prompts = [anno['video_caption'] for anno in annotations]
        elif self.prompt_type == 'action':
            prompts = [[anno['prompt']] for anno in annotations]
        elif self.prompt_type == 'llm_plan':
            prompts = [[anno['step_descriptions']] for anno in annotations]
        elif self.prompt_type == 'mix':
            prompts = [[anno['llm_caption']] + anno['image_caption'] for anno in annotations]
        else:
            raise ValueError("Invalid prompt type.")

        prompts = sum(prompts, []) + self.extra_prompts
        return list(set(prompts))


def skill_collate_fn(batch: list[dict]) -> dict:
    """
    Collect function for the dataloader.
    :param batch: list of dicts with keys:
        images: torch.tensor [N C H W]
        prompts: list[str]
        metadata: list[dict]
    :return: dict
        images: torch.tensor [B N C H W]
        prompts: list[list[str]]
        metadata: list[list[dict]]
        max_steps: int
        steps: list[int]
        prompt_embeds: list[torch.tensor]
        pooled_prompt_embeds: list[torch.tensor]
    """
    steps = [len(d['prompts']) for d in batch]
    max_step = max(steps)
    batch_size = len(batch)
    batch_images = torch.zeros(batch_size, max_step, *batch[0]['images'].shape[1:], dtype=batch[0]['images'].dtype)
    for i, d in enumerate(batch):
        batch_images[i, :len(d['prompts'])] = d['images']

    batch_prompts = [d['prompts'] + [''] * (max_step - len(d['prompts'])) for d in batch]

    return {
        'images':    batch_images,
        'prompts':   batch_prompts,
        'metadata':  [d['metadata'] for d in batch],
        'max_steps': max_step,
        'steps':     steps,
    }
