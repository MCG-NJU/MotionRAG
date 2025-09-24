# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import numpy as np
import torch
from einops import rearrange

from torch import nn, Tensor


class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/facebookresearch/llama/blob/main/llama/model.py#L450

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        num_head (int): Number of heads in the attention module.
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = None,
            max_seq_len: int = 4096,
            base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init()

    # We need to explicitly define reset_parameters for FSDP initialization, see
    # https://github.com/pytorch/pytorch/blob/797d4fbdf423dd9320ebe383fb57ffb1135c4a99/torch/distributed/fsdp/_init_utils.py#L885
    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (
                self.base
                ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def _forward(self, x: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [bsz, seq_len, num_heads, head_dim]
            input_pos (Optional[Tensor]): Optional tensor which contains the position
                of the current token. This is only used during inference. Default is None

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        TODO: The implementation below can be made more efficient
        for inference.
        """

        # input tensor has shape [b, s, n_h, n_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not. When
        # input_pos is provided, we're in inference mode
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, n_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [1, s, 1, n_d // 2, 2]
        rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, n_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, n_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)

    def forward(self, x: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        if len(x.shape) == 3:
            assert self.num_heads is not None, "num_heads must be set if 3D input is provided"
            x = rearrange(x, 'b s (n_h h_d) -> b s n_h h_d', n_h=self.num_heads)
            assert x.size(3) == self.dim, f"Expected dim to be {self.dim}, got {x.shape[-1]}"

            return rearrange(self._forward(x, input_pos), 'b s n_h h_d -> b s (n_h h_d)', n_h=self.num_heads)

        elif len(x.shape) == 4:
            return self._forward(x, input_pos)

        else:
            raise ValueError(f"Unsupported input dimension: {len(x.shape)}, expected 3D or 4D tensor")


class SinusoidPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int, max_length: int):
        super().__init__()
        self.dim = dim
        self.max_length = max_length
        self.pos_table = self.get_sinusoid_encoding_table(max_length, dim)

    # sin-cos position encoding
    # https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
    @staticmethod
    def get_sinusoid_encoding_table(n_position, d_hid):
        """Sinusoid position encoding table"""

        # TODO: make it with torch instead of numpy
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        assert x.size(-2) <= self.max_length, f'seq_len {x.size(-2)} > max_len {self.max_length}'
        return x + self.pos_table[:, :x.size(-2)].type_as(x).to(x.device).clone().detach()
