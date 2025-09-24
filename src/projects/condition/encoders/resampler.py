# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
# and https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py
# and https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/resampler.py
import math
import torch
import torch.nn as nn
from einops import rearrange


class ImageProjModel(nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        # embeds = image_embeds
        embeds = image_embeds.type(list(self.proj.parameters())[0].dtype)
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens,
                                                              self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class VideoProjModel(nn.Module):
    def __init__(self, cross_attention_dim=1024, action_embeddings_dim=1024, context_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.action_embeddings_dim = action_embeddings_dim
        self.context_tokens = context_tokens
        self.proj = nn.Linear(action_embeddings_dim, context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, action_embeds: torch.Tensor) -> torch.Tensor:
        action_embeds = action_embeds.mean(dim=1)
        latents = self.proj(action_embeds).reshape(-1, self.context_tokens, self.cross_attention_dim)
        return self.norm(latents)


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        # optimized attention
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.heads)
        out = nn.functional.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h l d -> b l (h d)')

        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(
            self,
            dim=1024,
            depth=8,
            dim_head=64,
            heads=16,
            num_queries=8,
            embedding_dim=768,
            output_dim=1024,
            ff_mult=4,
            video_length=None,  # using frame-wise version or not
            with_cls_token=False,
            ckpt_path=None,
    ):
        super().__init__()
        ## queries for a single frame / image
        self.num_queries = num_queries
        self.video_length = video_length
        self.with_cls_token = with_cls_token
        self.dim = dim
        self.embedding_dim = embedding_dim
        self.cross_attention_dim = output_dim
        self.output_dim = output_dim

        ## <num_queries> queries for each frame
        if video_length is not None:
            num_queries = num_queries * video_length
        if with_cls_token:
            num_queries += 1

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim ** 0.5)
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )
        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)

    def forward(self, x, return_cls_tokens=False):
        latents = self.latents.repeat(x.size(0), 1, 1)  ## B (T L) C
        x = self.proj_in(x)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        latents = self.norm_out(latents)  # B L C or B (T L) C

        if return_cls_tokens:
            assert self.with_cls_token is True, "with_cls_token must be True if return_cls_tokens is True"
            return latents[:, 0], latents[:, 1:]
        elif not return_cls_tokens and self.with_cls_token:
            return latents[:, 1:]
        else:
            return latents


class ActionProjModel(nn.Module):
    def __init__(self,
                 dim: int = 1024,
                 depth: int = 2,
                 embedding_shape: tuple[int, int, int] = (8, 14, 14),
                 embedding_dim=768,
                 output_shape: tuple[int, int, int] = (2, 4, 4),
                 ):
        """

        :param dim: dim of mlp
        :param depth: number of mlp layers
        :param embedding_shape: embedding shape (T H W)
        :param embedding_dim: embedding dim
        :param output_shape: output shape (T H W)
        """
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.embedding_shape = embedding_shape
        self.embedding_dim = embedding_dim
        self.output_shape = output_shape
        self.cross_attention_dim = embedding_dim

        self.sampler = nn.AdaptiveAvgPool3d(self.output_shape)

        self.mlp = [nn.Linear(self.embedding_dim, self.dim)]
        for _ in range(self.depth):
            self.mlp.append(nn.GELU())
            self.mlp.append(nn.Linear(self.dim, self.dim))
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, action_embeds: torch.Tensor) -> torch.Tensor:
        t, h, w = self.embedding_shape

        action_embeds = rearrange(action_embeds, 'b (t h w) c -> b c t h w', t=t, h=h, w=w)
        pooled_embeds = self.sampler(action_embeds)
        pooled_embeds = rearrange(pooled_embeds, 'b c t h w -> b (t h w) c')
        return self.mlp(pooled_embeds)


if __name__ == '__main__':
    resampler = Resampler(1024, 4, 64, 12, 16, 768, 1024, 4, 16)
    resampler = resampler.to('cuda', torch.bfloat16)

    from torch.utils.benchmark import Timer, Compare

    results = []
    with torch.no_grad():
        for batch in (1, 2, 3, 4):
            x = torch.randn(batch, 1536 * 3, 768, dtype=torch.bfloat16, device='cuda')

            results.append(Timer(stmt='out = resampler.forward(x)',
                                 globals={'resampler': resampler, 'x': x},
                                 sub_label=f'batch: {batch}',
                                 description='forward').blocked_autorange(min_run_time=1))

            results.append(Timer(stmt='out = resampler.efficient_forward(x)',
                                 globals={'resampler': resampler, 'x': x},
                                 sub_label=f'batch: {batch}',
                                 description='efficient_forward').blocked_autorange(min_run_time=1))

        Compare(results).print()
