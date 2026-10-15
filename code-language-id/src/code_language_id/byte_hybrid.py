"""Byte-level hybrid classifier (ByteHybrid).

Vendored verbatim from PleIAs/CommonLingua (Apache-2.0):
    https://huggingface.co/PleIAs/CommonLingua

Operates directly on raw UTF-8 bytes — no tokenizer required:

    raw bytes -> byte-embed + trigram-hash-embed (summed)
              -> N x depthwise Conv1D
              -> M x bidirectional attention (RoPE)
              -> masked mean-pool
              -> classification head

The single shipped configuration ("base_ngram") has d_model=256, 3 conv blocks,
1 attention block, 4 heads, kernel=15, 4096 trigram-hash buckets x 64 dim.
Total parameters ~2.35 M with 334 classes at max_len=512. The same architecture
is reused here for programming-language identification.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ByteNgramEmbed(nn.Module):
    """Rolling polynomial hash of byte trigrams into a fixed-size table."""

    def __init__(self, num_buckets: int = 4096, embed_dim: int = 64, n: int = 3) -> None:
        super().__init__()
        self.n = n
        self.num_buckets = num_buckets
        self.embed = nn.Embedding(num_buckets, embed_dim)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        B, T = byte_ids.shape
        clamped = byte_ids.clamp(max=255)
        padded = F.pad(clamped, (0, self.n - 1), value=0)
        h = torch.zeros(B, T, dtype=torch.long, device=byte_ids.device)
        for i in range(self.n):
            h = h * 257 + padded[:, i:i + T]
        return self.embed(h % self.num_buckets)


class ByteConvBlock(nn.Module):
    """Causal depthwise Conv1D + SwiGLU FFN, with residual + layernorm."""

    def __init__(self, d_model: int, kernel_size: int = 15, expand: int = 2) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model)
        self.norm2 = nn.LayerNorm(d_model)
        ffn = d_model * expand
        self.ffn_gate = nn.Linear(d_model, ffn, bias=False)
        self.ffn_up = nn.Linear(d_model, ffn, bias=False)
        self.ffn_down = nn.Linear(ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x).transpose(1, 2)
        x = F.pad(x, (self.pad, 0))
        x = F.silu(self.conv(x)).transpose(1, 2)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn_down(F.silu(self.ffn_gate(x)) * self.ffn_up(x))
        return residual + x


def _rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = q.shape[-1]
    seq_len = q.shape[-2]
    freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=q.device).float() / head_dim))
    t = torch.arange(seq_len, device=q.device)
    a = torch.outer(t, freqs)
    cos = a.cos().to(q.dtype)
    sin = a.sin().to(q.dtype)

    def rot(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    return rot(q), rot(k)


class ByteAttnBlock(nn.Module):
    """Bidirectional self-attention with RoPE + SwiGLU FFN."""

    def __init__(self, d_model: int, n_heads: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        ffn = d_model * expand
        self.ffn_gate = nn.Linear(d_model, ffn, bias=False)
        self.ffn_up = nn.Linear(d_model, ffn, bias=False)
        self.ffn_down = nn.Linear(ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        residual = x
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
        q, k = _rope(q, k)
        # Fused SDPA: avoids materializing the BxHxTxT attention matrix.
        # Math is identical to softmax(QK^T/sqrt(d))V; weights & checkpoints
        # remain compatible with the upstream manual implementation.
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=False
        )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        x = residual + self.out_proj(out)

        residual = x
        h = self.norm2(x)
        h = self.ffn_down(F.silu(self.ffn_gate(h)) * self.ffn_up(h))
        return residual + h


class ByteHybrid(nn.Module):
    """Byte-level classifier with optional trigram-hash augmentation.

    ``pool_type``:
      * ``"mean"`` — masked mean over the sequence (CommonLingua default; the
        only mode used by the 107-class lite checkpoint).
      * ``"attention"`` — learned softmax pool. A scalar attention score is
        produced per position by ``Linear(d_model, 1)``; softmax over unmasked
        positions yields the pooling weights. The projection is zero-init so
        the layer is exactly equivalent to mean pool at construction (uniform
        softmax over the unmasked positions), making warm-starting from a
        mean-pool checkpoint a no-op until training shapes the scores.
    """

    def __init__(
        self,
        num_classes: int,
        d_model: int = 256,
        n_conv: int = 3,
        n_attn: int = 1,
        n_heads: int = 4,
        ffn_expand: int = 2,
        max_len: int = 512,
        conv_kernel: int = 15,
        ngram_buckets: int = 0,
        ngram_dim: int = 64,
        pool_type: str = "mean",
    ) -> None:
        super().__init__()
        if pool_type not in {"mean", "attention"}:
            raise ValueError(f"pool_type must be 'mean' or 'attention', got {pool_type!r}")
        self.max_len = max_len
        self.pool_type = pool_type

        self.embed = nn.Embedding(257, d_model, padding_idx=256)

        self.ngram_embed = None
        if ngram_buckets > 0:
            self.ngram_embed = ByteNgramEmbed(ngram_buckets, ngram_dim, n=3)
            self.ngram_proj = nn.Linear(ngram_dim, d_model, bias=False)

        self.conv_layers = nn.ModuleList(
            [ByteConvBlock(d_model, conv_kernel, ffn_expand) for _ in range(n_conv)]
        )
        self.attn_layers = nn.ModuleList(
            [ByteAttnBlock(d_model, n_heads, ffn_expand) for _ in range(n_attn)]
        )
        self.final_norm = nn.LayerNorm(d_model)

        if pool_type == "attention":
            self.attn_pool_proj = nn.Linear(d_model, 1, bias=True)
            # Zero-init: scores=0 => softmax uniform over unmasked positions
            # => identical to mean pool at construction.
            nn.init.zeros_(self.attn_pool_proj.weight)
            nn.init.zeros_(self.attn_pool_proj.bias)
        else:
            self.attn_pool_proj = None

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        pad_mask = byte_ids != 256
        x = self.embed(byte_ids)
        if self.ngram_embed is not None:
            x = x + self.ngram_proj(self.ngram_embed(byte_ids))
        for layer in self.conv_layers:
            x = layer(x)
        for layer in self.attn_layers:
            x = layer(x)
        x = self.final_norm(x)
        if self.pool_type == "attention":
            scores = self.attn_pool_proj(x).squeeze(-1)              # [B, T]
            scores = scores.masked_fill(~pad_mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1).unsqueeze(-1)    # [B, T, 1]
            x = (x * weights).sum(dim=1)
        else:
            mask = pad_mask.unsqueeze(-1).to(x.dtype)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(x)


CONFIGS: dict[str, dict] = {
    "base_ngram": dict(
        d_model=256, n_conv=3, n_attn=1, n_heads=4, conv_kernel=15,
        ngram_buckets=4096, ngram_dim=64,
    ),
    "large_ngram": dict(
        d_model=384, n_conv=4, n_attn=2, n_heads=6, conv_kernel=15,
        ngram_buckets=8192, ngram_dim=96,
    ),
}
