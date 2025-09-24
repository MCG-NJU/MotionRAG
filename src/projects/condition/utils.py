from typing import Literal, Iterable

import torch
from einops import rearrange


def condition_fusion(
        condition_emb: torch.Tensor,
        fusion_type: Literal['mean', 'concat', 'top1', 'weight'] = 'mean',
        weight: torch.Tensor | Iterable[float] = None,
) -> torch.Tensor:
    """
    Fusion condition embedding on dim 1
    :param condition_emb: Tensor[b k l c]
    :param fusion_type: fusion type
    :param weight: weight for each condition
    :return: Tensor[b l c]
    """
    assert fusion_type in ['mean', 'concat', 'top1', 'weight']
    assert condition_emb.dim() == 4

    if fusion_type == 'mean':
        condition_emb = condition_emb.mean(dim=1)

    elif fusion_type == 'weight':
        distance: torch.Tensor = torch.tensor(weight, device=condition_emb.device)
        weight = (1 - distance) / (1 - distance).sum(dim=1, keepdim=True)
        condition_emb = (condition_emb * weight[..., None, None]).sum(dim=1)

    elif fusion_type == 'concat':
        condition_emb = rearrange(condition_emb, 'b k t c -> b (k t) c')

    elif fusion_type is None or fusion_type == 'top1':
        condition_emb = condition_emb[:, 0]

    return condition_emb


def extract_resampler_weights(input_ckpt_path, output_ckpt_path, resampler_prefix='resampler', strip_prefix=True):
    """
    Load resampler weights from a checkpoint and save as an individual checkpoint.

    Args:
        input_ckpt_path (str): Path to the input checkpoint file
        output_ckpt_path (str): Path to save the resampler weights checkpoint
        resampler_prefix (str, optional): Prefix that identifies resampler weights. Defaults to 'resampler'.
        strip_prefix (bool, optional): Whether to strip the prefix from the keys. Defaults to True.
    """
    # Load the checkpoint
    ckpt = torch.load(input_ckpt_path, map_location='cpu')

    # Extract model state_dict from checkpoint format
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model_state_dict = ckpt['model']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model_state_dict = ckpt['state_dict']
    else:
        # Assume the checkpoint itself is the state dict
        model_state_dict = ckpt

    # Extract resampler weights
    resampler_weights = {}
    for key, value in model_state_dict.items():
        # Check if the key contains the resampler prefix
        if resampler_prefix in key:
            if strip_prefix:
                # Remove the prefix and the following dot if present
                new_key = key.replace(f"{resampler_prefix}.", "")
                resampler_weights[new_key] = value.clone()
            else:
                resampler_weights[key] = value.clone()

    if not resampler_weights:
        print(f"Warning: No weights found with prefix '{resampler_prefix}'")
        print("Available keys:", list(model_state_dict.keys())[:10], "...")
        return

    # Save the resampler weights to a new checkpoint
    torch.save(resampler_weights, output_ckpt_path)

    print(f"Resampler weights extracted and saved to {output_ckpt_path}")
    print(f"Number of resampler weights: {len(resampler_weights)}")

    return resampler_weights
