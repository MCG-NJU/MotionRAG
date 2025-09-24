from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from ..basics import zero_module
from ..common import (
    checkpoint,
    exists,
    default,
)


class RelativePosition(nn.Module):
    """ https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py """

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        device = self.embeddings_table.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.long()
        embeddings = self.embeddings_table[final_mat]
        return embeddings


class CrossAttention(nn.Module):

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.,
                 relative_position=False, temporal_length=None, video_length=None, image_cross_attention=False,
                 image_cross_attention_scale=1.0, image_cross_attention_scale_learnable=False, text_context_len=77,
                 action_cross_attention=False, action_cross_attention_scale=1.0,
                 action_cross_attention_scale_learnable=False, mix_attention=False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        self.mix_attention = mix_attention
        if not mix_attention:
            self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
            self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        else:
            self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
            self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
            self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

        self.relative_position = relative_position
        if self.relative_position:
            assert (temporal_length is not None)
            self.relative_position_k = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
            self.relative_position_v = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
        else:
            self.forward = self.efficient_forward

        self.video_length = video_length
        self.image_cross_attention = image_cross_attention
        self.image_cross_attention_scale = image_cross_attention_scale
        self.action_cross_attention = action_cross_attention
        self.action_cross_attention_scale = action_cross_attention_scale
        self.text_context_len = text_context_len
        self.image_cross_attention_scale_learnable = image_cross_attention_scale_learnable
        self.action_cross_attention_scale_learnable = action_cross_attention_scale_learnable

        if self.image_cross_attention:
            self.to_k_ip = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_ip = nn.Linear(context_dim, inner_dim, bias=False)
            if image_cross_attention_scale_learnable:
                self.register_parameter('alpha', nn.Parameter(torch.tensor(0.)))

        if self.action_cross_attention:
            self.to_q_a = nn.Linear(inner_dim, inner_dim, bias=False)
            self.to_k_a = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_a = nn.Linear(context_dim, inner_dim, bias=False)
            if action_cross_attention_scale_learnable:
                self.register_parameter('alpha_action', nn.Parameter(torch.tensor(0.)))

    def forward(self, x, context=None, mask=None):
        raise NotImplementedError
        spatial_self_attn = (context is None) if not self.mix_attention else True
        k_ip, v_ip, out_ip = None, None, None

        h = self.heads
        q = self.to_q(x)
        if spatial_self_attn:
            k = self.to_k(x)
            v = self.to_v(x)
        else:
            k = self.to_k(context['prompt'])
            v = self.to_v(context['prompt'])

        if self.image_cross_attention:
            k_ip = self.to_k_ip(context['image'])
            v_ip = self.to_v_ip(context['image'])

        if self.action_cross_attention:
            k_a = self.to_k_a(context['action'])
            v_a = self.to_v_a(context['action'])

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        if self.relative_position:
            len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
            k2 = self.relative_position_k(len_q, len_k)
            sim2 = einsum('b t d, t s d -> b t s', q, k2) * self.scale  # TODO check
            sim += sim2
        del k

        if exists(mask):
            ## feasible for causal attention mask only
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b i j -> (b h) i j', h=h)
            sim.masked_fill_(~(mask > 0.5), max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        if self.relative_position:
            v2 = self.relative_position_v(len_q, len_v)
            out2 = einsum('b t s, t s d -> b t d', sim, v2)  # TODO check
            out += out2
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        ## for image cross-attention
        if self.image_cross_attention:
            k_ip, v_ip = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k_ip, v_ip))
            sim_ip = torch.einsum('b i d, b j d -> b i j', q, k_ip) * self.scale
            del k_ip
            sim_ip = sim_ip.softmax(dim=-1)
            out_ip = torch.einsum('b i j, b j d -> b i d', sim_ip, v_ip)
            out_ip = rearrange(out_ip, '(b h) n d -> b n (h d)', h=h)

            if self.image_cross_attention_scale_learnable:
                out = out + self.image_cross_attention_scale * out_ip * (torch.tanh(self.alpha) + 1)
            else:
                out = out + self.image_cross_attention_scale * out_ip

        if self.action_cross_attention:
            k_a, v_a = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k_a, v_a))
            sim_a = torch.einsum('b i d, b j d -> b i j', q, k_a) * self.scale
            del k_a
            sim_a = sim_a.softmax(dim=-1)
            out_a = torch.einsum('b i j, b j d -> b i d', sim_a, v_a)
            out_a = rearrange(out_a, '(b h) n d -> b n (h d)', h=h)

            if self.action_cross_attention_scale_learnable:
                out = out + self.action_cross_attention_scale * out_a * (torch.tanh(self.alpha_action) + 1)
            else:
                out = out + self.action_cross_attention_scale * out_a

        return self.to_out(out)

    def efficient_forward(self, x, context=None, mask=None):
        if exists(mask):
            raise NotImplementedError

        spatial_self_attn = (context is None) if not self.mix_attention else True

        q = self.to_q(x)
        if spatial_self_attn:
            k = self.to_k(x)
            v = self.to_v(x)
        else:
            k = self.to_k(context['prompt'])
            v = self.to_v(context['prompt'])

        q, k, v = map(
            lambda t: rearrange(t, 'b l (h d) -> b h l d', h=self.heads),
            (q, k, v),
        )
        out = nn.functional.scaled_dot_product_attention(q, k, v)

        if self.image_cross_attention:
            k_ip = self.to_k_ip(context['image'])
            v_ip = self.to_v_ip(context['image'])

            k_ip, v_ip = map(
                lambda t: rearrange(t, 'b l (h d) -> b h l d', h=self.heads),
                (k_ip, v_ip),
            )
            out_ip = nn.functional.scaled_dot_product_attention(q, k_ip, v_ip)

            if self.image_cross_attention_scale_learnable:
                out = out + self.image_cross_attention_scale * out_ip * (torch.tanh(self.alpha) + 1)
            else:
                out = out + self.image_cross_attention_scale * out_ip

        if self.action_cross_attention:
            q_a = self.to_q_a(rearrange(out, 'b h l d -> b l (h d)'))
            k_a = self.to_k_a(context['action'])
            v_a = self.to_v_a(context['action'])

            q_a, k_a, v_a = map(
                lambda t: rearrange(t, 'b l (h d) -> b h l d', h=self.heads),
                (q_a, k_a, v_a),
            )
            out_a = nn.functional.scaled_dot_product_attention(q_a, k_a, v_a)

            if self.action_cross_attention_scale_learnable:
                out = out + self.action_cross_attention_scale * out_a * (torch.tanh(self.alpha_action) + 1)
            else:
                out = out + self.action_cross_attention_scale * out_a

        out = rearrange(out, 'b h l d -> b l (h d)')
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):

    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False, attention_cls=None, video_length=None, image_cross_attention=False,
                 image_cross_attention_scale=1.0, image_cross_attention_scale_learnable=False, text_context_len=77,
                 action_cross_attention=False, action_cross_attention_scale_learnable=False,
                 ):
        super().__init__()
        attn_cls = CrossAttention if attention_cls is None else attention_cls
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              video_length=video_length, image_cross_attention=image_cross_attention,
                              image_cross_attention_scale=image_cross_attention_scale,
                              image_cross_attention_scale_learnable=image_cross_attention_scale_learnable,
                              text_context_len=text_context_len, action_cross_attention=action_cross_attention,
                              action_cross_attention_scale_learnable=action_cross_attention_scale_learnable, )
        self.image_cross_attention = image_cross_attention

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, mask=None, **kwargs):
        ## implementation tricks: because checkpointing doesn't support non-tensor (e.g. None or scalar) arguments
        input_tuple = (x,)  ## should not be (x), otherwise *input_tuple will decouple x into multiple arguments
        if context is not None:
            input_tuple = (x, context)
        if mask is not None:
            forward_mask = partial(self._forward, mask=mask)
            return checkpoint(forward_mask, (x,), self.parameters(), self.checkpoint)
        return checkpoint(self._forward, input_tuple, self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, mask=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, mask=mask) + x
        x = self.attn2(self.norm2(x), context=context, mask=mask) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data in spatial axis.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None,
                 use_checkpoint=True, disable_self_attn=False, use_linear=False, video_length=None,
                 image_cross_attention=False, image_cross_attention_scale_learnable=False,
                 action_cross_attention=False, action_cross_attention_scale_learnable=False, ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        attention_cls = None
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim,
                disable_self_attn=disable_self_attn,
                checkpoint=use_checkpoint,
                attention_cls=attention_cls,
                video_length=video_length,
                image_cross_attention=image_cross_attention,
                image_cross_attention_scale_learnable=image_cross_attention_scale_learnable,
                action_cross_attention=action_cross_attention,
                action_cross_attention_scale_learnable=action_cross_attention_scale_learnable,
            ) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None, **kwargs):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context, **kwargs)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class TemporalTransformer(nn.Module):
    """
    Transformer block for image-like data in temporal axis.
    First, reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None,
                 use_checkpoint=True, use_linear=False, only_self_att=True, causal_attention=False, causal_block_size=1,
                 relative_position=False, temporal_length=None, action_cross_attention=False,
                 action_cross_attention_scale_learnable=False):
        super().__init__()
        self.only_self_att = only_self_att
        self.relative_position = relative_position
        self.causal_attention = causal_attention
        self.causal_block_size = causal_block_size
        self.action_cross_attention = action_cross_attention
        self.action_cross_attention_scale_learnable = action_cross_attention_scale_learnable

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        if not use_linear:
            self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        if relative_position:
            assert (temporal_length is not None)
            attention_cls = partial(CrossAttention, relative_position=True, temporal_length=temporal_length,
                                    mix_attention=action_cross_attention)
        else:
            attention_cls = partial(CrossAttention, temporal_length=temporal_length,
                                    mix_attention=action_cross_attention)
        if self.causal_attention:
            assert (temporal_length is not None)
            self.mask = torch.tril(torch.ones([1, temporal_length, temporal_length]))

        if self.only_self_att and not action_cross_attention:
            context_dim = None
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim,
                attention_cls=attention_cls,
                action_cross_attention=action_cross_attention,
                action_cross_attention_scale_learnable=action_cross_attention_scale_learnable,
                checkpoint=use_checkpoint) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context: dict = None):
        b, c, t, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = rearrange(x, 'b c t h w -> (b h w) c t').contiguous()
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'bhw c t -> bhw t c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)

        temp_mask = None
        if self.causal_attention:
            # slice the from mask map
            temp_mask = self.mask[:, :t, :t].to(x.device)

        if temp_mask is not None:
            mask = temp_mask.to(x.device)
            mask = repeat(mask, 'l i j -> (l bhw) i j', bhw=b * h * w)
        else:
            mask = None

        if self.only_self_att:
            if not self.action_cross_attention:
                ## note: if no context is given, cross-attention defaults to self-attention
                for i, block in enumerate(self.transformer_blocks):
                    x = block(x, mask=mask)
            else:
                ctx = context.copy()
                ctx['action'] = rearrange(ctx['action'], '(b t) l c -> b t l c', t=t)[:, 0]
                ctx['action'] = repeat(ctx['action'], 'b l c -> (b hw) l c', hw=h * w)
                for i, block in enumerate(self.transformer_blocks):
                    x = block(x, context=ctx, mask=mask)
        else:
            for i, block in enumerate(self.transformer_blocks):
                ctx = context.copy()
                ctx['action'] = rearrange(ctx['action'], '(b t) l c -> b t l c', t=t)[:, 0]
                ctx['action'] = repeat(ctx['action'], 'b l c -> (b hw) l c', hw=h * w)
                x = block(x, context=ctx)

        x = rearrange(x, '(b hw) t c -> b hw t c', b=b).contiguous()

        if self.use_linear:
            x = self.proj_out(x)
            x = rearrange(x, 'b (h w) t c -> b c t h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = rearrange(x, 'b hw t c -> (b hw) c t').contiguous()
            x = self.proj_out(x)
            x = rearrange(x, '(b h w) c t -> b c t h w', b=b, h=h, w=w).contiguous()

        return x + x_in


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x + h_


if __name__ == '__main__':
    attn = CrossAttention(1024, 1024, 16, 64, temporal_length=16, image_cross_attention=True,
                          action_cross_attention=True, mix_attention=False)
    attn = attn.to('cuda', torch.bfloat16)

    from torch.utils.benchmark import Timer, Compare

    results = []
    with torch.no_grad():
        for batch in (1, 2, 3, 4):
            x = torch.randn(batch * 16, 1536, 1024, dtype=torch.bfloat16, device='cuda')
            ctx = {'action': torch.randn(batch * 16, 16, 1024, dtype=torch.bfloat16, device='cuda'),
                   'image':  torch.randn(batch * 16, 16, 1024, dtype=torch.bfloat16, device='cuda'),
                   'prompt': torch.randn(batch * 16, 77, 1024, dtype=torch.bfloat16, device='cuda')}

            results.append(Timer(stmt='out = attn.forward(x, context=ctx)',
                                 globals={'attn': attn, 'x': x, 'ctx': ctx},
                                 sub_label=f'batch: {batch}',
                                 description='forward').blocked_autorange(min_run_time=1))

            results.append(Timer(stmt='out = attn.efficient_forward(x, context=ctx)',
                                 globals={'attn': attn, 'x': x, 'ctx': ctx},
                                 sub_label=f'batch: {batch}',
                                 description='efficient_forward').blocked_autorange(min_run_time=1))
        Compare(results).print()
