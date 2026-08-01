"""
attention.py — TFDMGA Attention Mechanisms
============================================
Implements all attention modules used by the TFDMGA architecture:

  * MultiHeadSelfAttention     — per-modality MHSA with Flash Attention
  * CrossModalAttention        — cross-modality Q/K/V attention
  * TransformerEncoderBlock    — standard Pre-LN transformer block

All attention computations route through
``torch.nn.functional.scaled_dot_product_attention`` which automatically
selects the Flash Attention 2 CUDA kernel on Ampere/Ada GPUs (CUDA 12+,
PyTorch 2.0+), falling back to the efficient memory-efficient kernel or the
default math kernel as appropriate.

Author: TFDMGA Research Framework
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Utility: Pre-LayerNorm wrapper ──────────────────────────────────────────

class PreNorm(nn.Module):
    """Apply LayerNorm before a sub-module (Pre-LN formulation).

    Pre-LN training is substantially more stable than Post-LN for deep
    transformer stacks, especially with large learning rates.

    Parameters
    ----------
    dim : int
        Normalised feature dimension.
    fn : nn.Module
        The sub-module to wrap (attention or feed-forward).
    """

    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:  # type: ignore[override]
        return self.fn(self.norm(x), **kwargs)


# ─── Feed-Forward Network (used inside transformer blocks) ───────────────────

class FeedForward(nn.Module):
    """Position-wise two-layer feed-forward network with GELU activation.

    Architecture:
        Linear(dim → ff_dim) → GELU → Dropout → Linear(ff_dim → dim) → Dropout

    Parameters
    ----------
    dim : int
        Input and output feature dimension.
    ff_mult : int
        Hidden dimension multiplier relative to ``dim`` (default 4 → standard).
    dropout : float
        Dropout probability applied after the hidden layer and the output projection.
    """

    def __init__(self, dim: int, ff_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        ff_dim = dim * ff_mult
        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


# ─── Multi-Head Self-Attention ───────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention using Flash Attention when available.

    Internally projects input to Q, K, V using a single fused linear layer,
    reshapes to per-head tensors, calls
    ``F.scaled_dot_product_attention`` (which dispatches to Flash Attention 2
    on compatible hardware), then projects back to ``d_model``.

    Parameters
    ----------
    d_model : int
        Total model dimension. Must be divisible by ``n_heads``.
    n_heads : int
        Number of parallel attention heads.
    dropout : float
        Attention dropout probability (applied to the weight matrix).
    bias : bool
        Whether to include bias terms in projection layers.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        is_causal: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.dropout = dropout
        self.is_causal = is_causal

        # Fused Q, K, V projection for efficiency
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-head self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, T, d_model)`` where ``T`` is the
            sequence length (typically 1 for the cross-sectional financial
            data after encoding, or the number of time steps if temporal).
        attn_mask : Optional[torch.Tensor]
            Optional boolean or additive mask ``(B, T, T)`` or ``(T, T)``.

        Returns
        -------
        out : torch.Tensor
            Attention output of shape ``(B, T, d_model)``.
        attn_weights : torch.Tensor
            Averaged attention weight matrix of shape ``(B, T, T)``.
            Computed only when not in training mode (for interpretability).
        """
        B, T, _ = x.shape

        # Project and split into Q, K, V — shape (B, T, 3·d_model)
        qkv = self.qkv(x)
        # Reshape to (B, n_heads, T, head_dim) for each of Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Flash Attention / SDPA — automatically selects the best kernel
        attn_drop_p = self.dropout if self.training else 0.0
        is_causal = self.is_causal and attn_mask is None
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=attn_drop_p,
            is_causal=is_causal,
        )  # (B, n_heads, T, head_dim)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out)

        # Compute averaged attention weights for interpretability (no grad needed)
        with torch.no_grad():
            # Manual softmax-scaled attention for weight extraction
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q.detach(), k.detach().transpose(-2, -1)) * scale
            if attn_mask is not None:
                scores = scores + attn_mask
            elif is_causal:
                mask = torch.triu(torch.full((T, T), float('-inf'), device=scores.device), diagonal=1)
                scores = scores + mask
            attn_weights = torch.softmax(scores, dim=-1).mean(dim=1)  # avg over heads

        return out, attn_weights


# ─── Cross-Modal Attention ───────────────────────────────────────────────────

class CrossModalAttention(nn.Module):
    """Cross-modal attention: one modality queries another's keys and values.

    Implements the mechanism:
        out = Attention(Q=query_modal, K=context_modal, V=context_modal)

    This forces the query modality to selectively draw information from the
    context modality, learning inter-modal dependencies in a structured way.

    Architecture:
        - Separate linear projections for Q (query) and K,V (context)
        - ``F.scaled_dot_product_attention`` (Flash Attention when available)
        - Output projection back to ``d_model``
        - LayerNorm on the output

    Parameters
    ----------
    d_model : int
        Model dimension for both input modalities.
    n_heads : int
        Number of attention heads.
    dropout : float
        Attention dropout probability.
    bias : bool
        Include bias in projection layers.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        is_causal: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.is_causal = is_causal

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cross-modal attention.

        Parameters
        ----------
        query : torch.Tensor
            Query modality tensor ``(B, Tq, d_model)``.
        context : torch.Tensor
            Context modality (source of K, V) ``(B, Tc, d_model)``.

        Returns
        -------
        out : torch.Tensor
            Cross-attended output of shape ``(B, Tq, d_model)``.
            Includes a residual connection from ``query`` and LayerNorm.
        cross_attn_weights : torch.Tensor
            Averaged cross-attention weights ``(B, Tq, Tc)``.
        """
        B, Tq, _ = query.shape
        _, Tc, _ = context.shape

        # Project queries from query modality
        q = self.q_proj(query).view(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)

        # Project keys and values from context modality
        kv = self.kv_proj(context)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(B, Tc, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Tc, self.n_heads, self.head_dim).transpose(1, 2)

        attn_drop_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=attn_drop_p,
            is_causal=self.is_causal,
        )  # (B, n_heads, Tq, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        out = self.out_proj(out)

        # Residual + norm
        out = self.norm(out + query)

        # Extract averaged cross-attention weights (no grad)
        with torch.no_grad():
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q.detach(), k.detach().transpose(-2, -1)) * scale
            if self.is_causal:
                mask = torch.triu(torch.full((Tq, Tc), float('-inf'), device=scores.device), diagonal=1)
                scores = scores + mask
            cross_attn_weights = torch.softmax(scores, dim=-1).mean(dim=1)

        return out, cross_attn_weights


# ─── Transformer Encoder Block ───────────────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    """Full Pre-LN Transformer encoder block.

    Architecture (Pre-LN):
        x → LayerNorm → MHSA → + x  →  LayerNorm → FFN → + x

    Parameters
    ----------
    d_model : int
        Model dimension.
    n_heads : int
        Number of attention heads.
    ff_mult : int
        Feed-forward expansion factor (default 4).
    dropout : float
        Dropout applied in FFN and after attention.
    attention_dropout : float
        Attention-weight dropout in MHSA.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        is_causal: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=attention_dropout,
            is_causal=is_causal,
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ff = FeedForward(dim=d_model, ff_mult=ff_mult, dropout=dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through one transformer block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor ``(B, T, d_model)``.
        attn_mask : Optional[torch.Tensor]
            Optional attention mask.

        Returns
        -------
        x : torch.Tensor
            Output tensor ``(B, T, d_model)``.
        attn_weights : torch.Tensor
            Self-attention weights ``(B, T, T)`` for interpretability.
        """
        # Pre-LN self-attention with residual
        residual = x
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, attn_mask=attn_mask)
        x = residual + self.drop1(attn_out)

        # Pre-LN feed-forward with residual
        residual = x
        x = residual + self.drop2(self.ff(self.norm2(x)))

        return x, attn_weights


# ─── Self-contained smoke test ────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running attention smoke test on: {device}")

    B, T, D, H = 8, 30, 256, 8  # batch=8, seq_len=30 (temporal window size)

    # ── MHSA ─────────────────────────────────────────────────────────────────
    mhsa = MultiHeadSelfAttention(d_model=D, n_heads=H, dropout=0.1).to(device)
    x = torch.randn(B, T, D, device=device)
    out, weights = mhsa(x)
    assert out.shape == (B, T, D), f"MHSA out shape mismatch: {out.shape}"
    assert weights.shape == (B, T, T), f"MHSA weight shape mismatch: {weights.shape}"
    print(f"  MHSA     : out={tuple(out.shape)}, weights={tuple(weights.shape)} OK")
    
    # ── Cross-Modal Attention ─────────────────────────────────────────────────
    cma = CrossModalAttention(d_model=D, n_heads=H, dropout=0.1).to(device)
    ctx = torch.randn(B, T, D, device=device)
    out_cma, cw = cma(x, ctx)
    assert out_cma.shape == (B, T, D)
    print(f"  CrossAttn: out={tuple(out_cma.shape)}, weights={tuple(cw.shape)} OK")

    # ── Transformer Encoder Block ─────────────────────────────────────────────
    blk = TransformerEncoderBlock(d_model=D, n_heads=H, dropout=0.1).to(device)
    out_blk, aw = blk(x)
    assert out_blk.shape == (B, T, D)
    print(f"  TXBlock  : out={tuple(out_blk.shape)}, weights={tuple(aw.shape)} OK")

    print("All attention module tests passed.")
