---
license: mit
task_categories:
- image-to-video
- text-to-video
language:
- en
---

# 🤗 MotionRAG Model Checkpoints

## 📋 Overview

MotionRAG is a retrieval-augmented framework for image-to-video generation that significantly enhances motion realism by transferring motion priors from relevant reference videos. Our approach addresses the fundamental challenge of generating physically plausible and semantically coherent motion in video generation.

<p align="center">
  <img src="../assets/architecture-1.jpg" width="100%" alt="MotionRAG Framework Overview"/>
</p>

Our model checkpoints are organized into three key components for each base model:

1. **Motion Projector (Resampler)**: Compresses high-dimensional motion features from the video encoder into compact token representations.

2. **Motion Context Transformer**: Adapts motion patterns through in-context learning using a causal transformer architecture.

3. **Motion-Adapter**: Injects the adapted motion features into the base image-to-video generation models.

## 📦 Checkpoint Files

### MotionRAG Enhanced Models

| Model             | Component                  | File                                                |
|-------------------|----------------------------|-----------------------------------------------------|
| **CogVideoX**     | Fine-tuned for short video | `checkpoints/CogVideoX/17_frames.ckpt`              |
| **CogVideoX**     | Motion Projector           | `checkpoints/CogVideoX/motion_proj.ckpt`            |
| **CogVideoX**     | Motion Context Transformer | `checkpoints/CogVideoX/motion_transformer.ckpt`     |
| **CogVideoX**     | Motion-Adapter             | `checkpoints/CogVideoX/Motion-Adapter.ckpt`         |
| **DynamiCrafter** | Motion Projector           | `checkpoints/DynamiCrafter/motion_proj.ckpt`        |
| **DynamiCrafter** | Motion Context Transformer | `checkpoints/DynamiCrafter/motion_transformer.ckpt` |
| **DynamiCrafter** | Motion-Adapter             | `checkpoints/DynamiCrafter/Motion-Adapter.ckpt`     |
| **SVD**           | Motion Projector           | `checkpoints/SVD/motion_proj.ckpt`                  |
| **SVD**           | Motion Context Transformer | `checkpoints/SVD/motion_transformer.ckpt`           |
| **SVD**           | Motion-Adapter             | `checkpoints/SVD/Motion-Adapter.ckpt`               |

### Datasets

Our dataset differs from [OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M) datasets through curation and preprocessing. We use Llama3.1 to refine captions and extract motion-specific descriptions, which are stored in the `motion_caption` field. The data is then partitioned into non-overlapping training and test sets.

| Dataset        | Description                            | File                                          |
|----------------|----------------------------------------|-----------------------------------------------|
| **OpenVid-1M** | Large-scale video dataset for training | `datasets/OpenVid-1M/data/openvid-1m.parquet` |
| **OpenVid-1K** | Test set sampled from OpenVid-1M       | `datasets/OpenVid-1M/data/openvid-1k.parquet` |

## 🚀 Usage

For detailed usage instructions, please refer to the official repository: https://github.com/MCG-NJU/MotionRAG

## 📝 Citation

If you use these models in your research, please cite our paper:

```
@article{MotionRAG2025,
  title={MotionRAG: Motion Retrieval-Augmented Image-to-Video Generation},
  author={Hippocampus, David S.},
  journal={Advances in Neural Information Processing Systems},
  year={2025},
  url={https://arxiv.org/abs/2509.02813}, 
}
```

## 📬 Contact

For questions or issues related to the models, please open an issue on the repository.