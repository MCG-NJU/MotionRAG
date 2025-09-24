import json
import multiprocessing
import random
from functools import partial
from pathlib import Path
from typing import Callable, Literal
from typing import Tuple

import lightning.pytorch as pl
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .clip_selector import BaseSelector
from .dataset import VideoDataset, SkillImageDataset
from .rag import RAGDatabase


class VideoDataModule(pl.LightningDataModule):
    def __init__(self,
                 video_size: Tuple[int, int],
                 video_length: int,
                 train_clip_selector: BaseSelector,
                 val_clip_selector: BaseSelector,
                 test_clip_selector: BaseSelector,
                 train_annotation_path: list[str],
                 val_annotation_path: list[str],
                 test_annotation_path: list[str],
                 video_dir: str,
                 train_transforms,
                 read_video_fn: Literal['av', 'ta', 'tv'] = 'av',
                 num_workers: int = 4,
                 prefetch_factor: int = 4,
                 train_batch_size=32,
                 val_batch_size=32,
                 test_batch_size=32,
                 collate_fn: Callable = None,
                 tokenizer_model: str = "openai/clip-vit-base-patch16",
                 sampling_config: dict[int, float] = None,
                 uncond_text_ratio: float = 0.15,
                 uncond_video_ratio: float = 0.15,
                 use_ref_frame: bool = False,
                 ref_frame_dir: str | Path = None,
                 prompt_type: Literal["llm", "image", "video", "action", "llm_plan", "mix"] = 'llm',
                 ref_video_type: Literal["gt", "rag_text", "rag_text_image", "random"] = None,
                 ref_video_num: int = 1,
                 rag_prompt_type: Literal["image", "video", "action", "llm", "motion"] = 'llm',
                 rag_db_path: str | Path = 'data/rag.db',
                 annotation_cache_dir: str | Path = None,
                 dataset: Dataset = None,
                 ):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.video_size = video_size
        self.video_length = video_length
        self.train_clip_selector = train_clip_selector
        self.train_annotation_path = train_annotation_path
        self.val_clip_selector = val_clip_selector
        self.val_annotation_path = val_annotation_path
        self.test_clip_selector = test_clip_selector
        self.test_annotation_path = test_annotation_path
        self.video_dir = video_dir
        self.train_transforms = train_transforms if train_transforms else ()
        self.read_video_fn = read_video_fn
        self.tokenizer_model = tokenizer_model
        self.sampling_config = sampling_config
        self.uncond_text_ratio = uncond_text_ratio
        self.uncond_video_ratio = uncond_video_ratio
        self.prompt_type = prompt_type
        self.use_ref_frame = use_ref_frame
        self.ref_frame_dir = ref_frame_dir
        self.ref_video_type = ref_video_type
        self.ref_video_num = ref_video_num

        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.collate_fn = collate_fn

        self.annotation_cache_dir = Path(annotation_cache_dir) if annotation_cache_dir is not None else Path('/tmp')
        self.rag_database = None
        self.rag_db_path = rag_db_path
        self.rag_prompt_type = {'image':  'image_caption',
                                'video':  'video_caption',
                                'action': 'prompt',
                                'llm':    'llm_caption',
                                'motion': 'motion_caption'}[rag_prompt_type]

        self.dataset = VideoDataset if dataset is None else dataset

    def prepare_data(self) -> None:
        self.annotation_cache_dir.mkdir(exist_ok=True, parents=True)

        # move rag retrieval from dataset to data module
        if self.ref_video_type in ('rag_text', 'rag_text_image'):
            self.rag_database = RAGDatabase(self.rag_db_path, self.rag_prompt_type, 'cuda')

        if self.annotation_cache_dir == Path('/tmp') or not (self.annotation_cache_dir / 'train.pt').exists():
            self.prepare_annotations(self.train_annotation_path, self.train_clip_selector,
                                     save_path=self.annotation_cache_dir / 'train.pt',
                                     ref_video_type=self.ref_video_type,
                                     ref_video_num=self.ref_video_num)

        if self.annotation_cache_dir == Path('/tmp') or not (self.annotation_cache_dir / 'val.pt').exists():
            self.prepare_annotations(self.val_annotation_path, self.val_clip_selector,
                                     save_path=self.annotation_cache_dir / 'val.pt',
                                     ref_video_type=self.ref_video_type,
                                     ref_video_num=self.ref_video_num)
        if self.annotation_cache_dir == Path('/tmp') or not (self.annotation_cache_dir / 'test.pt').exists():
            self.prepare_annotations(self.test_annotation_path, self.test_clip_selector,
                                     save_path=self.annotation_cache_dir / 'test.pt',
                                     ref_video_type=self.ref_video_type,
                                     ref_video_num=self.ref_video_num)

        if self.rag_database is not None:
            del self.rag_database

    def setup(self, stage=None):
        self.train_dataset = self.dataset(
            annotation_path=self.annotation_cache_dir / 'train.pt',
            video_size=self.video_size,
            video_length=self.video_length,
            video_dir=self.video_dir,
            extra_transforms=self.train_transforms,
            read_video_fn=self.read_video_fn,
            tokenizer_model=self.tokenizer_model,
            sampling_config=self.sampling_config,
            uncond_text_ratio=self.uncond_text_ratio,
            uncond_video_ratio=self.uncond_video_ratio,
            use_ref_frame=self.use_ref_frame,
            ref_frame_dir=self.ref_frame_dir,
            prompt_type=self.prompt_type,
            ref_video_num=self.ref_video_num,
        )
        self.val_dataset = self.dataset(
            annotation_path=self.annotation_cache_dir / 'val.pt',
            video_size=self.video_size,
            video_length=self.video_length,
            video_dir=self.video_dir,
            extra_transforms=(),
            read_video_fn=self.read_video_fn,
            tokenizer_model=self.tokenizer_model,
            sampling_config={8: 1.0},
            uncond_text_ratio=0,
            uncond_video_ratio=0,
            use_ref_frame=self.use_ref_frame,
            ref_frame_dir=self.ref_frame_dir,
            prompt_type=self.prompt_type,
            ref_video_num=self.ref_video_num,
        )
        self.test_dataset = self.dataset(
            annotation_path=self.annotation_cache_dir / 'test.pt',
            video_size=self.video_size,
            video_length=self.video_length,
            video_dir=self.video_dir,
            extra_transforms=(),
            read_video_fn=self.read_video_fn,
            tokenizer_model=self.tokenizer_model,
            sampling_config={8: 1.0},
            uncond_text_ratio=0,
            uncond_video_ratio=0,
            use_ref_frame=self.use_ref_frame,
            ref_frame_dir=self.ref_frame_dir,
            prompt_type=self.prompt_type,
            ref_video_num=self.ref_video_num,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def prepare_annotations(self,
                            annotation_path: list[str],
                            clip_selector: BaseSelector,
                            save_path: Path,
                            ref_video_type: Literal["gt", "rag_text", "rag_text_image", "random"] = None,
                            ref_video_num: int = 1,
                            ) -> None:

        load_annotation_fn = partial(self.load_annotation, clip_selector=clip_selector, ref_video_type=ref_video_type,
                                     rag_prompt_type=self.rag_prompt_type)
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=max(len(annotation_path), 1)) as pool:
            annotations: list[list[dict]] = pool.map(load_annotation_fn, annotation_path, chunksize=1)
        annotations: list[dict] = sum(annotations, [])

        if ref_video_type is not None:
            if ref_video_type == 'gt':
                assert ref_video_num == 1, "ref_video_num must be 1 when ref_video_type is GT"
                fn = list_dict
                kwargs = [{'video':     anno['video'],
                           'start_sec': anno['start_sec'],
                           'end_sec':   anno['end_sec'],
                           '_distance': 0} for anno in annotations]

            elif ref_video_type == 'rag_text':
                fn = partial(dict2kwargs, self.rag_database.text_search)
                kwargs = [{'text':   anno['text_embedding'],
                           'top_k':  ref_video_num + 3,
                           'where':  f'video != "{anno["video"]}"',
                           'select': ['video', 'start_sec', 'end_sec']} for anno in annotations]


            elif ref_video_type == 'rag_text_image':
                fn = partial(dict2kwargs, self.rag_database.text_image_search)
                kwargs = [{'text':            anno['text_embedding'],
                           'image_embedding': anno['image_embedding'],
                           'top_k':           (ref_video_num * 2 + 3, ref_video_num),
                           'where':           f'video != "{anno["video"]}"',
                           'select':          ['video', 'start_sec', 'end_sec']} for anno in annotations]

            elif ref_video_type == 'random':
                fn = list
                kwargs = [[{'video':     anno['video'],
                            'start_sec': anno['start_sec'],
                            'end_sec':   anno['end_sec'],
                            '_distance': 0} for anno in random.choices(annotations, k=ref_video_num + 3)]
                          for _ in annotations]
            else:
                raise ValueError("Invalid ref_video_type.")

            ctx = multiprocessing.get_context('spawn')
            with ctx.Pool(processes=min(64, len(kwargs) // 128 + 1)) as pool:
                results: list[list[dict]] = list(tqdm(pool.imap(fn, kwargs, chunksize=128),
                                                      total=len(kwargs),
                                                      desc='Calculating ref videos',
                                                      leave=False))

            for anno, result in zip(annotations, results):
                anno['ref_videos']: list[dict] = result

        else:
            pass

        torch.save(annotations, save_path)

    @staticmethod
    def load_annotation(path: str | Path, clip_selector: BaseSelector,
                        ref_video_type: Literal['gt', 'rag_text', 'rag_text_image', 'random'] = None,
                        rag_prompt_type: str = None) -> list[dict]:
        """
        Load annotation from path and select clips by clip_selector
        :param path: Path to the annotation file.
        :param clip_selector: Clip selector.
        :param ref_video_type: Type of reference
        :param rag_prompt_type: Prompt for RAG
        :return:
        """
        path = Path(path)
        if path.suffix == '.pt':
            anno = torch.load(path)
        elif path.suffix == '.json':
            anno = json.load(path.open('r'))
            anno = [dict(a) for a in anno]
        elif path.suffix == '.parquet':
            anno = pd.read_parquet(path).to_dict(orient='records')
        else:
            raise NotImplementedError(f'Invalid annotation file suffix: {path.suffix}')
        print(f'Loading {path.stem}')

        if ref_video_type in ('rag_text', 'rag_text_image'):
            anno_no_emb = [a for a in anno if 'text_embedding' not in a]
            text = [a[rag_prompt_type] if a[rag_prompt_type] is not None else '' for a in anno_no_emb]

            model = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)
            print('Calculating text embedding...')
            embeddings = model.encode(text, batch_size=256, show_progress_bar=True)
            for a, emb in zip(anno_no_emb, embeddings):
                a['text_embedding'] = emb

        if ref_video_type == 'rag_text_image' and 'image_embedding' not in anno[0]:
            image_emb = torch.load(f'data/eva_clip/{path.stem}.pt')
            assert len(anno) == len(image_emb), f'image_emb len {len(image_emb)} != anno len {len(anno)}'
            for a, emb in zip(anno, image_emb):
                assert a['video'] == emb['video'], 'image embedding not match'
                a['image_embedding'] = emb['eva_clip_vision']

            selected_anno = clip_selector(anno)
            for a in selected_anno:
                a['image_embedding'] = a['image_embedding'][0].numpy()
        else:
            selected_anno = clip_selector(anno)

        # filter out Tensor in annotation
        selected_anno = [{k: v for k, v in anno.items() if not isinstance(v, torch.Tensor)} for anno in selected_anno]
        return selected_anno


def dict2kwargs(func, dict):
    return func(**dict)


def list_dict(d: dict) -> list[dict]:
    return [d]


class SkillImageDataModule(VideoDataModule):
    def setup(self, stage=None):
        self.dataset = SkillImageDataset
        super().setup(stage)
