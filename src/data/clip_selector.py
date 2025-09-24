import random
from collections import defaultdict
from typing import Literal, Callable, Iterable

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d


class BaseSelector:
    def __call__(self, annotation: list[dict]) -> list[dict]:
        raise NotImplementedError


class AllSelector(BaseSelector):
    def __call__(self, annotation: list[dict]) -> list[dict]:
        return annotation


class IndexSelector(BaseSelector):
    def __init__(self,
                 indices: Iterable[int],
                 ):
        """
        Select clips based on indices.
        :param indices: Indices of clips to select.
        """
        self.indices = indices

    def __call__(self, annotation: list[dict]) -> list[dict]:
        return [annotation[i] for i in self.indices]


class RandomSelector(BaseSelector):
    def __init__(self,
                 num_clips: int = 1,
                 seed: int = 42,
                 ):
        """
        Select clips randomly.
        :param num_clips: Number of clips to select.
        :param seed: Seed for random number generator.
        """
        self.num_clips = num_clips
        self.seed = seed

    def __call__(self, annotation: list[dict]) -> list[dict]:
        assert self.num_clips <= len(
            annotation), f"num_clips ({self.num_clips}) must be less than or equal to the number of annotations ({len(annotation)})"
        random.seed(self.seed)
        return random.sample(annotation, self.num_clips)


class LengthSelector(BaseSelector):
    def __init__(self,
                 min_clip_len: float = 1,
                 max_clip_len: float = 10,
                 ):
        """
        Select clips based on length.
        :param min_clip_len: Minimum length of a clip in seconds.
        :param max_clip_len: Maximum length of a clip in seconds.
        """
        self.min_clip_len = min_clip_len
        self.max_clip_len = max_clip_len

    def __call__(self, annotation: list[dict]) -> list[dict]:
        clips = []
        for anno_id, anno in enumerate(annotation):
            if self.min_clip_len <= anno['end_sec'] - anno['start_sec'] <= self.max_clip_len:
                clips.append(anno)
        return clips


class ResolutionSelector(BaseSelector):
    def __init__(self,
                 min_resolution: tuple[int, int] = (540, 960),
                 ):
        """
        Select clips based on resolution.
        :param min_resolution: Threshold for resolution in (height, width).
        """
        self.min_resolution = min_resolution

    def __call__(self, annotation: list[dict]) -> list[dict]:
        return [anno for anno in annotation if anno.get('resolution', (0, 0)) >= self.min_resolution]


class ThresholdSelector(BaseSelector):
    def __init__(self,
                 metric_name: str,
                 goal: Literal['max', 'min', 'custom'] = 'min',
                 threshold: float | torch.Tensor = 1,
                 min_clip_len: float = 1,
                 ):
        """
        Select clips based on a metric.
        :param metric_name: Metric name.
        :param goal: Maximum or minimum value of metric.
        :param threshold: Threshold to select clips.
        :param min_clip_len: Minimum length of a clip in seconds.
        """
        self.metric_name = metric_name
        self.goal = goal
        self.threshold = threshold
        self.min_clip_len = min_clip_len

        if self.goal == 'max':
            self.is_good_enough = self.greater_than_threshold
        elif self.goal == 'min':
            self.is_good_enough = self.less_than_threshold
        elif self.goal == 'custom':
            pass
        else:
            raise ValueError(f"goal must be 'max' or 'min' or 'custom', but got {self.goal}")

    def greater_than_threshold(self, x):
        return x > self.threshold

    def less_than_threshold(self, x):
        return x < self.threshold

    def is_invalid_annotation(self, anno: dict) -> bool:
        """
        Check if an annotation is invalid.
        :param anno: Annotation to check.
        :return: True if the annotation is invalid.
        """
        return anno[self.metric_name] is None

    def is_good_enough(self, metric_value: torch.Tensor) -> torch.Tensor:
        """
        Check if a metric value is good enough.
        :param metric_value: Metric value.
        :return: True if the metric value is good enough.
        """
        raise NotImplementedError()

    def __call__(self, annotation: list[dict]) -> list[dict]:
        clips = []
        for anno_id, anno in enumerate(annotation):
            if self.is_invalid_annotation(anno):
                continue

            start_sec = round(anno['start_sec'] * anno['fps']) / anno['fps']
            idx = torch.where(~self.is_good_enough(anno[self.metric_name]))[0]
            idx = [-1, *idx.tolist(), len(anno[self.metric_name])]

            fps = anno['fps']
            min_frame = round(self.min_clip_len * fps)

            for start, end in zip(idx[:-1], idx[1:]):
                start += 1
                if end - start >= min_frame:
                    subclip_anno = anno.copy()
                    subclip_anno.update({
                        'start_sec': start_sec + start / fps,
                        'end_sec':   start_sec + end / fps,
                        **{k: v[start: end] for k, v in subclip_anno.items() if isinstance(v, torch.Tensor)}
                    })
                    clips.append(subclip_anno)
        return clips


class GaussianFilterSelector(ThresholdSelector):
    def __init__(self,
                 metric_name: str,
                 goal: Literal['max', 'min'] = 'min',
                 threshold: float = 1,
                 subclip_len: float = 1,
                 n_subclips: int = 1,
                 sigma_ratio: float = 4,
                 ):
        """
        Select clips based on the gaussian filtered metric.
        :param metric_name: Metric name.
        :param goal: Maximum or minimum value of metric.
        :param threshold: Threshold to select clips.
        :param subclip_len: Length of a subclip in seconds.
        :param n_subclips: Max number of subclips per clip.
        :param sigma_ratio: Ratio of sigma to subclip length. i.e. sigma = subclip_len // sigma_ratio
        """
        super().__init__(metric_name, goal, threshold, threshold)
        self.subclip_len = subclip_len
        self.n_subclips = n_subclips
        self.sigma_ratio = sigma_ratio

        if self.goal == 'max':
            self.padding_value = -np.inf
            self.is_good_enough = self.greater_than_threshold
            self.find_best_idx = np.argmax
        elif self.goal == 'min':
            self.padding_value = np.inf
            self.is_good_enough = self.less_than_threshold
            self.find_best_idx = np.argmin

    def find_best_idx(self, metric_values: np.ndarray) -> int:
        """
        Find the best index of an array of metric value.
        :param metric_values: Metric values.
        :return: Index of the best metric value.
        """
        raise NotImplementedError()

    def __call__(self, annotation: list[dict]) -> list[dict]:
        clips = []
        for anno_id, anno in enumerate(annotation):
            if self.is_invalid_annotation(anno):
                continue

            start_sec = round(anno['start_sec'] * anno['fps']) / anno['fps']
            end_sec = round(anno['end_sec'] * anno['fps']) / anno['fps']

            if end_sec - start_sec <= self.subclip_len:
                try:
                    metric_mean = anno[self.metric_name].mean().item()
                except AttributeError:
                    metric_mean = anno[self.metric_name]  # is a scalar
                if self.is_good_enough(metric_mean):
                    subclip_anno = anno.copy()
                    subclip_anno.update({
                        self.metric_name: metric_mean,
                        'start_sec':      start_sec,
                        'end_sec':        end_sec,
                    })
                    clips.append(subclip_anno)
            else:
                fps = anno['fps']
                clip_max_frames = round(self.subclip_len * fps)
                # make sure the frame number is odd
                clip_max_frames = clip_max_frames - 1 if clip_max_frames % 2 == 0 else clip_max_frames

                ma_value = gaussian_filter1d(anno[self.metric_name],
                                             sigma=clip_max_frames // self.sigma_ratio,
                                             mode='constant',
                                             cval=self.padding_value,
                                             radius=clip_max_frames // 2)

                for i in range(self.n_subclips):
                    best_idx = self.find_best_idx(ma_value)
                    best_value = float(ma_value[best_idx])
                    start, end = best_idx - clip_max_frames // 2, best_idx + clip_max_frames // 2

                    # early stop if no more clips can be selected
                    if not self.is_good_enough(best_value):
                        break

                    subclip_anno = anno.copy()
                    subclip_anno.update({
                        self.metric_name: best_value,
                        'start_sec':      start_sec + start / fps,
                        'end_sec':        start_sec + end / fps,
                        **{k: v[start: end] for k, v in subclip_anno.items() if
                           isinstance(v, torch.Tensor) and k != self.metric_name}
                    })
                    clips.append(subclip_anno)

                    # mask the selected area
                    ma_value[start: end] = self.padding_value

        # sanity check
        for c in clips:
            assert c['end_sec'] > c['start_sec']
            assert self.is_good_enough(c[self.metric_name])

        return clips


class SubClipSelector(BaseSelector):
    def __init__(self,
                 max_subclips: int = 3,
                 ):
        """
        Select given number of subclips from each clip based on their scores.
        """
        self.max_subclips = max_subclips

    def rank_fn(self, x: dict):
        """Choose which subclips to select. subclip with lower score will be selected."""
        return -x['clip_score']

    @staticmethod
    def gather_subclips_of_same_clip(subclips: list[dict]) -> list[list[dict]]:
        """
        Gather subclips of the same clip.
        :param subclips: Results.
        :return: A list of results of the same clip.
        """
        dict_of_subclips = defaultdict(list)
        for s in subclips:
            dict_of_subclips[s['id']].append(s)

        return [value for key, value in sorted(dict_of_subclips.items())]

    def __call__(self, annotation: list[dict]) -> list[dict]:
        assert all('id' in d for d in annotation), "Annotation must have 'id' attribute"
        grouped_anno = self.gather_subclips_of_same_clip(annotation)

        annotations = []
        for group in grouped_anno:
            annotations += sorted(group, key=self.rank_fn)[:self.max_subclips]

        return annotations


class CompositionSelector(BaseSelector):
    def __init__(self,
                 selectors: Iterable[BaseSelector],
                 ):
        """
        Select clips based on multiple selectors.
        :param selectors: Selectors to use.
        """
        self.selectors = tuple(selectors)

    def __call__(self, annotation: list[dict]) -> list[dict]:
        init_num_clips = len(annotation)

        for selector in self.selectors:
            annotation_filted = selector(annotation)
            print(
                f'{selector.__class__.__name__}: {len(annotation)}->{len(annotation_filted)}; {len(annotation_filted) / len(annotation) * 100:.2f}%')
            annotation = annotation_filted

        print(f'Total: {init_num_clips}->{len(annotation)}; {len(annotation) / init_num_clips * 100:.2f}%\n')

        return annotation


class SkillSelector(BaseSelector):
    def __init__(self,
                 min_steps: int = 2,
                 max_steps: int = 7,
                 remove_tensor_attr: bool = True,
                 ):
        """
        Gather all steps of a skill.
        :param min_steps: Minimum number of steps.
        :param max_steps: Maximum number of steps.
        :param remove_tensor_attr: Whether to remove tensor attributes.
        """
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.remove_tensor_attr = remove_tensor_attr

    def __call__(self, annotation: list[dict]) -> list[dict]:
        """
        Gather all steps of a skill and limits the number of steps.
        :param annotation: Annotations of single steps.
        :return: A list of all steps of a skill.
        """
        dict_of_skills = defaultdict(list)
        for anno in annotation:
            if self.remove_tensor_attr:
                anno = {k: v for k, v in anno.items() if not isinstance(v, torch.Tensor)}
            dict_of_skills[anno['video']].append(anno)

        outputs = []
        for vid, anno_list in dict_of_skills.items():
            if self.min_steps <= len(anno_list) <= self.max_steps:
                outputs.append({'steps': sorted(anno_list, key=lambda x: x['start_sec'])})

        return outputs


class SceneSelector(ThresholdSelector):
    def __init__(self,
                 scene_threshold: tuple[float, ...] = (0.5, 1.5, 2.5),
                 min_subclip_len: float = 1,
                 ):
        """
        Select clips based on scene changes.
        :param scene_threshold: Threshold for detecting scene changes.
        :param min_subclip_len: Minimum length of a clip in seconds.
        """
        super().__init__(metric_name='scene_score',
                         goal='custom',
                         threshold=torch.tensor(scene_threshold),
                         min_clip_len=min_subclip_len)

    def is_good_enough(self, metric_value: torch.Tensor) -> torch.Tensor:
        return torch.all(metric_value < self.threshold, dim=1)


class TextSelector(ThresholdSelector):
    def __init__(self,
                 text_threshold: float = 0.005,
                 min_subclip_len: float = 1,
                 ):
        """
        Select clips based on text area percent.
        :param text_threshold: Threshold for detecting text percent.
        :param min_subclip_len: Minimum length of a clip in seconds.
        """
        super().__init__(metric_name='text_score',
                         goal='min',
                         threshold=text_threshold,
                         min_clip_len=min_subclip_len)


class MotionSelector(GaussianFilterSelector):
    def __init__(self,
                 subclip_len: float = 3,
                 n_subclip: int = 2,
                 max_kl: float = 4,
                 ):
        """
        Select clips based on motion changes.
        :param subclip_len: Target length of a clip in seconds.
        :param n_subclip: Number of subclips to select from each clip.
        :param max_kl: Maximum KL divergence of subclips.
        """
        super().__init__(metric_name='motion_score',
                         goal='min',
                         threshold=max_kl,
                         subclip_len=subclip_len,
                         n_subclips=n_subclip, )


class SemanticsSelector(GaussianFilterSelector):
    def __init__(self,
                 subclip_len: float = 3,
                 n_subclip: int = 2,
                 min_similarity: float = 0.3,
                 ):
        """
        Select clips based on semantics similarity.
        :param subclip_len: Target length of a clip in seconds.
        :param n_subclip: Number of subclips to select from each clip.
        :param min_similarity: Minimum similarity of subclips.
        """
        super().__init__(metric_name='clip_score',
                         goal='max',
                         threshold=min_similarity,
                         subclip_len=subclip_len,
                         n_subclips=n_subclip, )


class SceneMotionSelector(CompositionSelector):
    def __init__(self,
                 scene_threshold: tuple[float, ...] = (0.5, 1.5, 2.5),
                 min_subclip_len: float = 1,
                 subclip_len: float = 3,
                 n_subclip: int = 2,
                 max_kl: float = 4,
                 ):
        """
        Select clips based on scene changes and motion changes.
        :param scene_threshold: Threshold for detecting scene changes.
        :param min_subclip_len: Minimum length of a clip in seconds.
        :param subclip_len: Target length of a clip in seconds.
        :param n_subclip: Number of subclips to select from each clip.
        :param max_kl: Maximum KL divergence of subclips.
        """
        self.scene_selector = SceneSelector(scene_threshold=scene_threshold, min_subclip_len=min_subclip_len)
        self.motion_selector = MotionSelector(subclip_len=subclip_len, n_subclip=n_subclip, max_kl=max_kl)
        self.subclip_selector = SubClipSelector(max_subclips=n_subclip)

        self.subclip_selector.rank_fn = self.rank_fn

        super().__init__((self.scene_selector, self.motion_selector, self.subclip_selector))

    @staticmethod
    def rank_fn(x: dict) -> float:
        return x['motion_score']


class SceneSemanticsMotionSelector(CompositionSelector):
    def __init__(self,
                 scene_threshold: tuple[float, ...] = (0.5, 1.5, 2.5),
                 min_subclip_len: float = 1,
                 subclip_len: float = 3,
                 n_subclip: int = 2,
                 min_similarity: float = 0.3,
                 max_kl: float = 4,
                 ):
        """
        Select clips based on scene changes and semantics similarity.
        :param scene_threshold: Threshold for detecting scene changes.
        :param min_subclip_len: Minimum length of a clip
        :param subclip_len: Target length of a clip in seconds.
        :param n_subclip: Number of subclips to select from each clip.
        :param min_similarity: Minimum similarity of subclips.
        :param max_kl: Maximum KL divergence of subclips.
        """
        self.scene_selector = SceneSelector(scene_threshold=scene_threshold, min_subclip_len=min_subclip_len)
        self.motion_selector = MotionSelector(subclip_len=subclip_len, n_subclip=n_subclip, max_kl=max_kl)
        self.semantics_selector = SemanticsSelector(subclip_len=subclip_len, n_subclip=n_subclip * 3,
                                                    min_similarity=min_similarity)
        self.subclip_selector = SubClipSelector(max_subclips=n_subclip)
        super().__init__((self.scene_selector, self.motion_selector, self.semantics_selector, self.subclip_selector))


class SceneMotionSemanticsSelector(CompositionSelector):
    def __init__(self,
                 scene_threshold: tuple[float, ...] = (0.5, 1.5, 2.5),
                 min_subclip_len: float = 1,
                 subclip_len: float = 3,
                 n_subclip: int = 2,
                 max_kl: float = 4,
                 min_similarity: float = 0.3,
                 ):
        """
        Select clips based on scene changes, motion changes and semantics similarity.
        :param scene_threshold: Threshold for detecting scene changes.
        :param min_subclip_len: Minimum length of a clip in seconds.
        :param subclip_len: Target length of a clip in seconds.
        :param n_subclip: Number of subclips to select from each clip.
        :param max_kl: Maximum KL divergence between subclips.
        :param min_similarity: Minimum similarity between subclips.
        """
        self.scene_selector = SceneSelector(scene_threshold=scene_threshold, min_subclip_len=min_subclip_len)
        self.motion_selector = MotionSelector(subclip_len=subclip_len, n_subclip=n_subclip * 3, max_kl=max_kl)
        self.semantics_selector = SemanticsSelector(subclip_len=subclip_len, n_subclip=n_subclip,
                                                    min_similarity=min_similarity)
        self.subclip_selector = SubClipSelector(max_subclips=n_subclip)

        super().__init__((self.scene_selector, self.motion_selector, self.semantics_selector, self.subclip_selector))


class SceneTextMotionSemanticsSelector(CompositionSelector):
    def __init__(self,
                 scene_threshold: tuple[float, ...] = (0.5, 1.5, 2.5),
                 text_threshold: float = 0.005,
                 min_subclip_len: float = 1,
                 subclip_len: float = 3,
                 n_subclip: int = 2,
                 max_kl: float = 4,
                 min_similarity: float = 0.3,
                 ):
        """
        Select clips based on scene changes, text area, motion changes and semantics similarity.
        :param scene_threshold: Threshold for detecting scene changes.
        :param text_threshold: Threshold for detecting text area.
        :param min_subclip_len: Minimum length of a clip in seconds.
        :param subclip_len: Target length of a clip in seconds.
        :param n_subclip: Number of subclips to select from each clip.
        :param max_kl: Maximum KL divergence between subclips.
        :param min_similarity: Minimum similarity between subclips.
        """
        self.scene_selector = SceneSelector(scene_threshold=scene_threshold, min_subclip_len=min_subclip_len)
        self.text_selector = TextSelector(text_threshold=text_threshold, min_subclip_len=min_subclip_len)
        self.motion_selector = MotionSelector(subclip_len=subclip_len, n_subclip=n_subclip * 3, max_kl=max_kl)
        self.semantics_selector = SemanticsSelector(subclip_len=subclip_len, n_subclip=n_subclip,
                                                    min_similarity=min_similarity)
        self.subclip_selector = SubClipSelector(max_subclips=n_subclip)

        super().__init__(
            (self.scene_selector, self.text_selector, self.motion_selector, self.semantics_selector,
             self.subclip_selector))


class ResolutionSceneTextMotionSemanticsSelector(CompositionSelector):
    def __init__(self,
                 scene_threshold: tuple[float, ...] = (0.5, 1.5, 2.5),
                 text_threshold: float = 0.005,
                 min_subclip_len: float = 1,
                 subclip_len: float = 3,
                 n_subclip: int = 2,
                 max_kl: float = 4,
                 min_similarity: float = 0.3,
                 min_resolution: tuple[int, int] = (540, 960),
                 ):
        """
        Select clips based on scene changes, text area, motion changes and semantics similarity.
        :param scene_threshold: Threshold for detecting scene changes.
        :param text_threshold: Threshold for detecting text area.
        :param min_subclip_len: Minimum length of a clip in seconds.
        :param subclip_len: Target length of a clip in seconds.
        :param n_subclip: Number of subclips to select from each clip.
        :param max_kl: Maximum KL divergence between subclips.
        :param min_similarity: Minimum similarity between subclips.
        :param min_resolution: Minimum resolution of a clip.
        """
        self.resolution_selector = ResolutionSelector(min_resolution=min_resolution)
        self.scene_selector = SceneSelector(scene_threshold=scene_threshold, min_subclip_len=min_subclip_len)
        self.text_selector = TextSelector(text_threshold=text_threshold, min_subclip_len=min_subclip_len)
        self.motion_selector = MotionSelector(subclip_len=subclip_len, n_subclip=n_subclip * 3, max_kl=max_kl)
        self.semantics_selector = SemanticsSelector(subclip_len=subclip_len, n_subclip=n_subclip,
                                                    min_similarity=min_similarity)
        self.subclip_selector = SubClipSelector(max_subclips=n_subclip)

        super().__init__(
            (self.resolution_selector, self.scene_selector, self.text_selector, self.motion_selector,
             self.semantics_selector, self.subclip_selector))


class SceneTextSemanticsSkillSelector(CompositionSelector):
    def __init__(self,
                 scene_threshold: tuple[float, ...] = (0.5, 1.5, 2.5),
                 text_threshold: float = 0.01,
                 min_subclip_len: float = 2,
                 subclip_len: float = 3,
                 n_subclip: int = 1,
                 min_similarity: float = 0.25,
                 min_steps: int = 2,
                 max_steps: int = 7,
                 ):
        """
        Select clips based on scene changes, text area, semantics similarity and skill steps.
        :param scene_threshold: Threshold for detecting scene changes.
        :param text_threshold: Threshold for detecting text area.
        :param min_subclip_len: Minimum length of a clip in seconds.
        :param subclip_len: Target length of a clip in seconds.
        :param n_subclip: Number of subclips to select from each clip.
        :param min_similarity: Minimum similarity between subclips and prompt.
        :param min_steps: Minimum number of steps in a skill.
        :param max_steps: Maximum number of steps in a skill.
        """
        self.scene_selector = SceneSelector(scene_threshold=scene_threshold, min_subclip_len=min_subclip_len)
        self.text_selector = TextSelector(text_threshold=text_threshold, min_subclip_len=min_subclip_len)
        self.semantics_selector = SemanticsSelector(subclip_len=subclip_len, n_subclip=n_subclip,
                                                    min_similarity=min_similarity)
        self.subclip_selector = SubClipSelector(max_subclips=n_subclip)

        self.skill_selector = SkillSelector(min_steps=min_steps, max_steps=max_steps)
        super().__init__((self.scene_selector, self.text_selector, self.semantics_selector, self.subclip_selector,
                          self.skill_selector))


class RandomSkillSelector(CompositionSelector):
    def __init__(self,
                 min_steps: int = 2,
                 max_steps: int = 7,
                 num_skill: int = 10,
                 seed: int = 7,
                 ):
        """
        Select clips based on skill steps.
        :param min_steps: Minimum number of steps in a skill.
        :param max_steps: Maximum number of steps in a skill.
        :param num_skill: Number of skills to select.
        :param seed: Seed for random selection.
        """
        self.skill_selector = SkillSelector(min_steps=min_steps, max_steps=max_steps)
        self.random_selector = RandomSelector(num_skill, seed)
        super().__init__([self.skill_selector, self.random_selector])


if __name__ == '__main__':
    data = torch.load('../../data/coin.pt')
    a = SceneTextMotionSemanticsSelector()
    a(data)
