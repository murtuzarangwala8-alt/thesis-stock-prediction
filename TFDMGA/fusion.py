"""
fusion.py — TFDMGA Fusion Components
======================================
Implements the modality fusion layer:

  * ModalEncoder          — independent encoder per modality (Tech / Fund / Macro)
  * DynamicGatingNetwork  — learns per-modality adaptive trust weights
  * ResidualFusionBlock   — merges gated representations with skip connections

Author: TFDMGA Research Framework
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Residual Block (used inside ModalEncoder) ────────────────────────────────

class ResidualBlock(nn.Module):
    """Pre-LN residual block: LayerNorm → Linear → GELU → Dropout → Linear → Dropout → + residual.

    If ``in_dim != out_dim``, a linear projection is applied to the residual
    branch so dimensions match for the skip connection.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output feature dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.fc1 = nn.Linear(in_dim, out_dim * 2)   # expand then contract (GLU-style)
        self.fc2 = nn.Linear(out_dim * 2, out_dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

        # Projection for residual if dimensions differ
        self.proj = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        residual = self.proj(x)
        h = self.norm(x)
        h = self.act(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return residual + h


# ─── Modal Encoder ───────────────────────────────────────────────────────────

class ModalEncoder(nn.Module):
    """Independent modality-specific encoder.

    Architecture (per encoder):
        Input → LayerNorm → Linear(in_dim → d_model) → GELU
              → [ResidualBlock × n_layers]
              → LayerNorm → Linear(d_model → d_model)
              → unsqueeze(1)   # add sequence dimension for attention

    The output has shape ``(B, 1, d_model)`` — a single dense token
    representing the entire modality. This token is subsequently passed
    through multi-head self-attention and cross-modal attention.

    Parameters
    ----------
    in_dim : int
        Number of raw input features for this modality.
    d_model : int
        Output latent dimension.
    n_layers : int
        Number of stacked residual blocks.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Stack of residual blocks — all operate at d_model dimension
        self.blocks = nn.ModuleList(
            [ResidualBlock(d_model, d_model, dropout=dropout) for _ in range(n_layers)]
        )

        self.output_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a raw modality feature vector.

        Parameters
        ----------
        x : torch.Tensor
            Raw features ``(B, in_dim)``.

        Returns
        -------
        torch.Tensor
            Encoded token ``(B, 1, d_model)`` ready for attention.
        """
        h = self.input_norm(x)
        h = self.input_proj(h)
        for block in self.blocks:
            h = block(h)
        h = self.output_norm(h)
        h = self.output_proj(h)
        return h.unsqueeze(1)  # (B, 1, d_model)


# ─── Dynamic Gating Network ──────────────────────────────────────────────────

class DynamicGatingNetwork(nn.Module):
    """Macro-conditioned dynamic gating network (Audit Fix C7).

    Computes per-modality adaptive trust weights conditioned SOLELY on
    the macroeconomic representation, implementing the thesis equation:

        w_t = softmax(W_g · h_t^{macro} / τ)

    The macro branch acts as a regime detector: during high-volatility
    macro regimes, the gates may upweight technical signals; during
    stable regimes, fundamental signals may dominate.

    The gating output is 3-way (tech, fund, sent) because macro itself
    does NOT contribute to the fusion sum — it only modulates the other
    modalities.

    Architecture:
        h_macro  →  LayerNorm → Linear → GELU → Dropout
          → Linear(d_model → 3)  →  softmax(·/τ)
        Output: [w_tech, w_fund, w_sent] ∈ (0,1)³,  Σ = 1

    Parameters
    ----------
    d_model : int
        Encoded representation dimension per modality.
    n_gated_modalities : int
        Number of modalities being gated (default 3: tech, fund, sent).
        Note: macro is the gating *signal*, not a gated modality.
    hidden_dim : int
        Hidden layer size in the gating MLP.
    dropout : float
        Dropout in the gating MLP.
    init_temperature : float
        Initial value of the learnable softmax temperature.
    """

    def __init__(
        self,
        d_model: int,
        n_gated_modalities: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        init_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_gated_modalities = n_gated_modalities
        # Gate input is ONLY macro (d_model), not all modalities concatenated
        in_dim = d_model

        self.gate_net = nn.Sequential(
            nn.LayerNorm(in_dim, eps=1e-6),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_gated_modalities),
        )
        # Learnable temperature — initialised near 1 for smooth softmax
        self.log_temperature = nn.Parameter(
            torch.tensor(float(init_temperature)).log()
        )

    def forward(
        self,
        h_tech: torch.Tensor,
        h_fund: torch.Tensor,
        h_macro: torch.Tensor,
        h_sent: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dynamic modality gates from macro context only.

        Parameters
        ----------
        h_tech : torch.Tensor
            Technical encoded token (unused for gating, kept for API compat).
        h_fund : torch.Tensor
            Fundamental encoded token (unused for gating).
        h_macro : torch.Tensor
            Macro encoded token — the SOLE conditioning signal ``(B, d_model)``
            or ``(B, T, d_model)``.
        h_sent : torch.Tensor
            Sentiment encoded token (unused for gating).

        Returns
        -------
        gates : torch.Tensor
            Soft gate weights ``(B, 3)`` summing to 1.
            Column order: [w_tech, w_fund, w_sent].
        """
        # Use ONLY macro as gating context (thesis Eq. 10)
        if h_macro.dim() == 3:
            hm = h_macro.mean(dim=1)  # (B, d_model)
        else:
            hm = h_macro

        # Macro-only context for gate computation
        ctx = hm  # (B, d_model) — NOT concatenated with other modalities
        logits = self.gate_net(ctx)  # (B, 3)

        # Temperature-scaled softmax; clamp temperature to avoid division by ~0
        temperature = self.log_temperature.exp().clamp(min=0.1)
        gates = F.softmax(logits / temperature, dim=-1)  # (B, 3)
        return gates


# ─── Residual Fusion Block ────────────────────────────────────────────────────

class ResidualFusionBlock(nn.Module):
    """Fuses gated modality representations via a residual feed-forward network.

    Input: weighted combination of the three modality tokens.
    Architecture:
        h_fused = w_tech · h_tech + w_fund · h_fund + w_macro · h_macro
                                                    # (B, d_model)
        out = LayerNorm(h_fused
              + Dropout(Linear(GELU(Linear(LayerNorm(h_fused))))))

    The residual skip connection preserves the raw gated signal while
    allowing the feed-forward network to learn higher-order interactions.

    Parameters
    ----------
    d_model : int
        Modality encoded dimension (input).
    fusion_dim : int
        Hidden expansion dimension in the fusion feed-forward.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        fusion_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Linear(d_model, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        h_tech: torch.Tensor,
        h_fund: torch.Tensor,
        h_macro: torch.Tensor,
        h_sent: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse gated modality representations (Audit Fix C7).

        Macro is EXCLUDED from the weighted sum — it only acts as the
        conditioning signal for the gating network.  The fusion is:

            h_fused = w_tech · h_tech + w_fund · h_fund + w_sent · h_sent

        This implements thesis Equation 10: macro modulates but does not
        contribute a direct embedding to the final prediction.

        Parameters
        ----------
        h_tech, h_fund, h_macro, h_sent : torch.Tensor
            Post-attention tensors ``(B, T, d_model)`` or ``(B, d_model)``.
            h_macro is accepted for API compatibility but NOT used in fusion.
        gates : torch.Tensor
            Soft gate weights ``(B, T, 3)`` or ``(B, 3)`` from DynamicGatingNetwork.
            Column order: [w_tech, w_fund, w_sent].

        Returns
        -------
        torch.Tensor
            Fused representation ``(B, T, d_model)`` or ``(B, d_model)``.
        """
        # 3-way gates: [tech, fund, sent] — macro excluded from fusion
        if h_tech.dim() == 3:
            if gates.dim() == 3:
                w_tech = gates[:, :, 0:1]   # (B, T, 1)
                w_fund = gates[:, :, 1:2]
                w_sent = gates[:, :, 2:3]
            else:
                w_tech = gates[:, 0:1].unsqueeze(-1)   # (B, 1, 1)
                w_fund = gates[:, 1:2].unsqueeze(-1)
                w_sent = gates[:, 2:3].unsqueeze(-1)
        else:
            w_tech = gates[:, 0:1]   # (B, 1)
            w_fund = gates[:, 1:2]
            w_sent = gates[:, 2:3]

        # Weighted combination — macro is NOT included (acts only as gate signal)
        h = w_tech * h_tech + w_fund * h_fund + w_sent * h_sent  # (B, T, d_model) or (B, d_model)

        # Residual feed-forward
        h = self.norm2(h + self.ff(self.norm1(h)))
        return h


# ─── Self-contained smoke test ────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running fusion smoke test on: {device}")

    B, D = 16, 256
    tech_in, fund_in, macro_in, sent_in = 46, 192, 26, 2

    enc_tech  = ModalEncoder(tech_in,  D, n_layers=3).to(device)
    enc_fund  = ModalEncoder(fund_in,  D, n_layers=3).to(device)
    enc_macro = ModalEncoder(macro_in, D, n_layers=3).to(device)
    enc_sent  = ModalEncoder(sent_in,  D, n_layers=3).to(device)

    x_tech  = torch.randn(B, tech_in,  device=device)
    x_fund  = torch.randn(B, fund_in,  device=device)
    x_macro = torch.randn(B, macro_in, device=device)
    x_sent  = torch.randn(B, sent_in,  device=device)

    h_tech  = enc_tech(x_tech)   # (B, 1, D)
    h_fund  = enc_fund(x_fund)
    h_macro = enc_macro(x_macro)
    h_sent  = enc_sent(x_sent)

    assert h_tech.shape == (B, 1, D), f"Tech encoder output shape mismatch: {h_tech.shape}"
    print(f"  ModalEncoder (tech) : {tuple(h_tech.shape)} OK")
    print(f"  ModalEncoder (fund) : {tuple(h_fund.shape)} OK")
    print(f"  ModalEncoder (macro): {tuple(h_macro.shape)} OK")
    print(f"  ModalEncoder (sent) : {tuple(h_sent.shape)} OK")

    gate_net = DynamicGatingNetwork(d_model=D, n_gated_modalities=3).to(device)
    gates = gate_net(h_tech.squeeze(1), h_fund.squeeze(1), h_macro.squeeze(1), h_sent.squeeze(1))
    assert gates.shape == (B, 3), f"Gate shape mismatch: {gates.shape}"
    assert torch.allclose(gates.sum(dim=-1), torch.ones(B, device=device), atol=1e-5)
    print(f"  DynamicGating (macro-only): {tuple(gates.shape)} OK  (sum~1: {gates.sum(-1).mean():.4f})")

    fusion = ResidualFusionBlock(d_model=D, fusion_dim=512).to(device)
    fused = fusion(h_tech.squeeze(1), h_fund.squeeze(1), h_macro.squeeze(1), h_sent.squeeze(1), gates)
    assert fused.shape == (B, D)
    print(f"  ResidualFusionBlock : {tuple(fused.shape)} OK")

    # Sequence shape test
    h_tech_seq = torch.randn(B, 30, D, device=device)
    h_fund_seq = torch.randn(B, 30, D, device=device)
    h_macro_seq = torch.randn(B, 30, D, device=device)
    h_sent_seq = torch.randn(B, 30, D, device=device)
    gates_seq = gate_net(h_tech_seq, h_fund_seq, h_macro_seq, h_sent_seq)
    assert gates_seq.shape == (B, 3), f"Gate seq shape mismatch: {gates_seq.shape}"
    fused_seq = fusion(h_tech_seq, h_fund_seq, h_macro_seq, h_sent_seq, gates_seq)
    assert fused_seq.shape == (B, 30, D)
    print(f"  ResidualFusionBlock (Seq): {tuple(fused_seq.shape)} OK")

    print("All fusion module tests passed.")
