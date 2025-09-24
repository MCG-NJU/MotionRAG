from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0, Attention, CogVideoXAttnProcessor2_0
from einops import repeat
from torch import nn


class APAdapterAttnProcessor2_0(IPAdapterAttnProcessor2_0):
    def __init__(self, hidden_size, cross_attention_dim=None, num_tokens=(4,), scale=1.0):
        super().__init__(hidden_size, cross_attention_dim, num_tokens, scale)

        self.to_q_ip = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(len(num_tokens))]
        )

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            action_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            temb: Optional[torch.Tensor] = None,
            scale: float = 1.0,
            ip_adapter_masks: Optional[torch.Tensor] = None,
    ):
        """
        Modified from IPAdapterAttnProcessor2_0
        """
        residual = hidden_states
        # separate ip_hidden_states from encoder_hidden_states or action_hidden_states
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states
            else:
                assert action_hidden_states is not None, "action_hidden_states must be provided"
                ip_hidden_states = action_hidden_states

            ip_hidden_states = [ip_hidden_states]

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # for ip-adapter
        for current_ip_hidden_states, scale, to_q_ip, to_k_ip, to_v_ip in zip(
                ip_hidden_states, self.scale, self.to_q_ip, self.to_k_ip, self.to_v_ip
        ):
            skip = False
            if isinstance(scale, list):
                if all(s == 0 for s in scale):
                    skip = True
            elif scale == 0:
                skip = True
            if not skip:
                ip_query = to_q_ip(hidden_states)
                ip_key = to_k_ip(current_ip_hidden_states)
                ip_value = to_v_ip(current_ip_hidden_states)

                r = batch_size // current_ip_hidden_states.size(0)
                ip_key = repeat(ip_key, "b ... -> (b r) ...", r=r, b=ip_key.size(0))
                ip_value = repeat(ip_value, "b ... -> (b r) ...", r=r, b=ip_value.size(0))

                ip_query = ip_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.scale when we move to Torch 2.1
                current_ip_hidden_states = F.scaled_dot_product_attention(
                    ip_query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                )

                current_ip_hidden_states = current_ip_hidden_states.transpose(1, 2).reshape(
                    batch_size, -1, attn.heads * head_dim
                )
                current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)

                hidden_states = hidden_states + scale * current_ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class APAdapterCogVideoXAttnProcessor2_0(nn.Module):
    def __init__(self, hidden_size, cross_attention_dim=None, num_tokens=(4,), scale=1.0):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = [num_tokens]
        self.num_tokens = num_tokens

        if not isinstance(scale, list):
            scale = [scale] * len(num_tokens)
        if len(scale) != len(num_tokens):
            raise ValueError("`scale` should be a list of integers with the same length as `num_tokens`.")
        self.scale = scale

        self.to_k_ip = nn.ModuleList(
            [nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in range(len(num_tokens))]
        )
        self.to_v_ip = nn.ModuleList(
            [nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in range(len(num_tokens))]
        )
        self.to_q_ip = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(len(num_tokens))]
        )

    def __call__(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            action_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor]:
        """
        Modified from APAdapterAttnProcessor2_0
        """
        # separate ip_hidden_states from image_rotary_emb or action_hidden_states
        if isinstance(image_rotary_emb, tuple) and isinstance(image_rotary_emb[1], torch.Tensor):
            image_rotary_emb, ip_hidden_states = image_rotary_emb
        else:
            assert action_hidden_states is not None, "action_hidden_states must be provided"
            ip_hidden_states = action_hidden_states

        ip_hidden_states = [ip_hidden_states]

        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # for ip-adapter
        for current_ip_hidden_states, scale, to_q_ip, to_k_ip, to_v_ip in zip(
                ip_hidden_states, self.scale, self.to_q_ip, self.to_k_ip, self.to_v_ip
        ):
            skip = False
            if isinstance(scale, list):
                if all(s == 0 for s in scale):
                    skip = True
            elif scale == 0:
                skip = True
            if not skip:
                ip_query = to_q_ip(hidden_states)
                ip_key = to_k_ip(current_ip_hidden_states)
                ip_value = to_v_ip(current_ip_hidden_states)

                r = batch_size // current_ip_hidden_states.size(0)
                ip_key = repeat(ip_key, "b ... -> (b r) ...", r=r, b=ip_key.size(0))
                ip_value = repeat(ip_value, "b ... -> (b r) ...", r=r, b=ip_value.size(0))

                ip_query = ip_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.scale when we move to Torch 2.1
                current_ip_hidden_states = F.scaled_dot_product_attention(
                    ip_query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                )

                current_ip_hidden_states = current_ip_hidden_states.transpose(1, 2).reshape(
                    batch_size, -1, attn.heads * head_dim
                )
                current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)

                hidden_states = hidden_states + scale * current_ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states
