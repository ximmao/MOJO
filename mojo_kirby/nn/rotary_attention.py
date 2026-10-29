from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mojo_kirby.nn.rotary_embedding import apply_rotary_pos_emb


class RotaryCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        rotate_value: bool = False,
        batch_type: str = "stacked",
        backend: str = "math",
    ):
        super().__init__()

        inner_dim = dim_head * heads
        context_dim = context_dim or dim
        self.heads = heads
        self.dropout = dropout
        self.rotate_value = rotate_value

        if batch_type != "stacked":
            raise ValueError(f"Unknown batch_type: {batch_type}, must be 'stacked'")
        self.batch_type = batch_type

        if backend != "math":
            raise ValueError(f"Unknown backend: {backend}, must be 'math'")
        self.backend = backend

        # build networks
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x_query,
        x_context,
        query_pos_emb,
        context_pos_emb,
        *,
        context_mask=None,
        query_seqlen=None,
        context_seqlen=None,
    ):
        # normalize and project to q, k, v
        x_query = self.norm(x_query)
        x_context = self.norm_context(x_context)

        q = self.to_q(x_query)
        k, v = self.to_kv(x_context).chunk(2, dim=-1)

        assert query_seqlen is None and context_seqlen is None

        out = rotary_attn_func(
            query=q,
            key=k,
            value=v,
            q_pos_emb=query_pos_emb,
            kv_pos_emb=context_pos_emb,
            num_heads=self.heads,
            dropout_p=self.dropout if self.training else 0,
            rotate_value=self.rotate_value,
            attn_mask=context_mask,
        )

        # project back to dim
        out = self.to_out(out)
        return out


class RotarySelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        rotate_value: bool = False,
        batch_type: str = "stacked",
        backend: str = "math",
    ):
        super().__init__()

        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.rotate_value = rotate_value

        if batch_type != "stacked":
            raise ValueError(f"Unknown batch_type: {batch_type}, must be 'stacked'")
        self.batch_type = batch_type

        if backend != "math":
            raise ValueError(f"Unknown backend: {backend}, must be 'math'")
        self.backend = backend

        # build networks
        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(
        self,
        x,
        rotary_time_emb,
        *,
        x_mask=None,
        x_seqlen=None,
    ):

        if x_seqlen is not None:
            raise ValueError("x_seqlen must be None for stacked batching.")

        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        out = rotary_attn_func(
            query=q,
            key=k,
            value=v,
            q_pos_emb=rotary_time_emb,
            kv_pos_emb=rotary_time_emb,
            num_heads=self.heads,
            dropout_p=self.dropout if self.training else 0,
            rotate_value=self.rotate_value,
            attn_mask=x_mask,
        )

        out = self.to_out(out)
        return out


def rotary_attn_func(
    *,
    query,
    key,
    value,
    q_pos_emb,
    kv_pos_emb,
    attn_mask=None,
    num_heads: int,
    dropout_p: float,
    rotate_value: bool,
):

    # default attention expects shape b h n d
    query = rearrange(query, "b n (h d) -> b h n d", h=num_heads)
    key = rearrange(key, "b n (h d) -> b h n d", h=num_heads)
    value = rearrange(value, "b n (h d) -> b h n d", h=num_heads)

    # apply rotary embeddings
    query = apply_rotary_pos_emb(q_pos_emb, query, head_dim=1)
    key = apply_rotary_pos_emb(kv_pos_emb, key, head_dim=1)
    if rotate_value:
        value = apply_rotary_pos_emb(kv_pos_emb, value, head_dim=1)

    # attention mask
    if attn_mask is not None:
        if attn_mask.dim() == 2:  # attn_mask is (B, n_kv)
            attn_mask = rearrange(attn_mask, "b n -> b () () n")
        elif attn_mask.dim() == 3:  # attn_mask is (B, n_q, n_kv)
            attn_mask = rearrange(attn_mask, "b l s -> b () l s")
        elif attn_mask.dim() == 4:  # attn_mask is (B, h, n_q, n_kv)
            pass
        else:
            raise ValueError("unrecognized dimension")

    # perform attention, by default will use the optimal attention implementation
    out = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
    )

    if rotate_value:
        out = apply_rotary_pos_emb(-q_pos_emb, out, head_dim=1)

    # return (b, n, (h d), )
    out = rearrange(out, "b h n d -> b n (h d)")
    return out
