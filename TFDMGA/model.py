"""
model.py — TFDMGA Full Architecture
=====================================
Assembles all sub-components into the complete
Temporal Fusion Deep Multimodal Gated Attention Network (TFDMGA).

Forward pass overview
--------------------    x_tech  (B, 46)  ─► TechnicalEncoder  ─► (B, 1, D) ─► MHSA_tech  ─►┐
    x_fund  (B, 192) ─► FundamentalEncoder ─► (B, 1, D) ─► MHSA_fund  ─►├─► CrossModalAttention (3-way ring)
    x_macro (B, 26)  ─► MacroEncoder       ─► (B, 1, D) ─► MHSA_macro ─►┘
                                                                                 │
                                             DynamicGatingNetwork ◄─────────────—│
                                                      │ gates (B,3)            │
                                             ResidualFusionBlock               │
                                                      │
                                      TransformerEncoderBlock × N
                                                      │
                               ┌──────────┴──────────┬───────────┐
                           Head_1d (→1) Head_21d (→1) Head_126d (→1)
                              ↑                ↑              ↑
                           gate_tech        gate_macro     gate_fund
                           (daily trade)  (monthly trade) (6m trade)

Author: TFDMGA Research Framework
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CrossModalAttention, MultiHeadSelfAttention, TransformerEncoderBlock
from .config import TFDMGAConfig
from .fusion import DynamicGatingNetwork, ModalEncoder, ResidualFusionBlock
from .utils import count_parameters, log_model_summary, setup_logger


# ─── Prediction Head ─────────────────────────────────────────────────────────

class PredictionHead(nn.Module):
    """Two-layer MLP prediction head with BatchNorm and GELU.

    Architecture:
        LayerNorm → Linear(d → hidden) → GELU → Dropout → Linear(hidden → 1)

    Parameters
    ----------
    in_dim : int
        Input dimension (shared backbone output dimension).
    hidden_dim : int
        Hidden layer dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim, eps=1e-6),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


# ─── Causal & TCN Modules ───────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """Causal 1D Convolutional Layer that pads strictly on the left."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_padded = F.pad(x, (self.padding, 0))
        return self.conv(x_padded)


class CausalTCNBlock(nn.Module):
    """Residual causal convolutional block with BatchNorm evaluation fix."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(
            out_channels, out_channels, kernel_size, dilation=dilation
        )
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act_out = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        
        out = self.conv1(x)
        if out.shape[0] > 1 or not self.training:
            out = self.norm1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        if out.shape[0] > 1 or not self.training:
            out = self.norm2(out)
        out = self.act2(out)
        out = self.drop2(out)

        return self.act_out(out + residual)


class ModalityCausalTCN(nn.Module):
    """Causal Temporal Convolutional Network for a single modality."""
    def __init__(
        self,
        in_dim: int,
        d_model: int,
        tcn_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        layers = []
        curr_channels = in_dim
        channels = list(tcn_channels) + [d_model]
        
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(
                CausalTCNBlock(
                    in_channels=curr_channels,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            curr_channels = out_ch
            
        self.tcn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, T, D_in) -> transpose to (B, D_in, T)
        x = x.transpose(1, 2)
        out = self.tcn(x)
        # Output shape: (B, D_model, T) -> transpose to (B, T, D_model)
        return out.transpose(1, 2)


# ─── TFDMGA Main Model ───────────────────────────────────────────────────────

class TFDMGA(nn.Module):
    """Temporal Fusion Deep Multimodal Gated Attention Network.

    A novel multi-modal deep learning architecture for joint prediction of
    1-day and 21-day forward stock returns from heterogeneous financial signals.

    The model learns *how much to trust* each information modality (technical,
    fundamental, macro) dynamically per sample via a trainable gating network,
    and captures inter-modal dependencies through a structured cross-modal
    attention scheme before fusing everything in a stack of transformer blocks.

    Parameters
    ----------
    config : TFDMGAConfig
        Master configuration object (see ``config.py``).

    Inputs
    ------
    x_tech  : torch.Tensor  (B, tech_dim)
    x_fund  : torch.Tensor  (B, fund_dim)
    x_macro : torch.Tensor  (B, macro_dim)

    Outputs
    -------
    pred_1d   : torch.Tensor  (B, 1)   — predicted 1-day forward return
    pred_21d  : torch.Tensor  (B, 1)   — predicted 21-day forward return
    pred_126d : torch.Tensor  (B, 1)   — predicted 126-day (6-month) forward return
    aux       : Dict[str, torch.Tensor] — interpretability tensors:
                 "gates"          (B, 3)  modality trust weights [tech, fund, sent]
                 "gate_tech"      (B, 1)  technical gate weight
                 "gate_fund"      (B, 1)  fundamental gate weight
                 "gate_sent"      (B, 1)  sentiment gate weight
                 "attn_tech"      (B, 1, 1) self-attention weights (tech)
                 "attn_fund"      (B, 1, 1) self-attention weights (fund)
                 "attn_macro"     (B, 1, 1) self-attention weights (macro)
                 "cross_tf"       (B, 1, 1) Tech←Fund cross-attention
                 "cross_fm"       (B, 1, 1) Fund←Macro cross-attention
                 "cross_mt"       (B, 1, 1) Macro←Tech cross-attention
                 "transformer_aw" List[(B, 1, 1)] per-block attention weights
    """

    def __init__(self, config: TFDMGAConfig) -> None:
        super().__init__()
        self.config = config
        D = config.d_model

        # ── 1. Independent Modal Encoders ─────────────────────────────────────
        self.tech_encoder = ModalityCausalTCN(
            in_dim=config.tech_dim,
            d_model=D,
            tcn_channels=config.tcn_channels,
            kernel_size=config.tcn_kernel_size,
            dropout=config.dropout,
        )
        self.fund_encoder = ModalityCausalTCN(
            in_dim=config.fund_dim,
            d_model=D,
            tcn_channels=config.tcn_channels,
            kernel_size=config.tcn_kernel_size,
            dropout=config.dropout,
        )
        self.macro_encoder = ModalityCausalTCN(
            in_dim=config.macro_dim,
            d_model=D,
            tcn_channels=config.tcn_channels,
            kernel_size=config.tcn_kernel_size,
            dropout=config.dropout,
        )
        self.sent_encoder = ModalityCausalTCN(
            in_dim=config.sent_dim,
            d_model=D,
            tcn_channels=config.tcn_channels,
            kernel_size=config.tcn_kernel_size,
            dropout=config.dropout,
        )

        # ── 2. Per-Modal Multi-Head Self-Attention ────────────────────────────
        self.mhsa_tech = MultiHeadSelfAttention(
            d_model=D,
            n_heads=config.n_heads,
            dropout=config.attention_dropout,
        )
        self.mhsa_fund = MultiHeadSelfAttention(
            d_model=D,
            n_heads=config.n_heads,
            dropout=config.attention_dropout,
        )
        self.mhsa_macro = MultiHeadSelfAttention(
            d_model=D,
            n_heads=config.n_heads,
            dropout=config.attention_dropout,
        )
        self.mhsa_sent = MultiHeadSelfAttention(
            d_model=D,
            n_heads=config.n_heads,
            dropout=config.attention_dropout,
        )

        # ── 3. Cross-Modal Attention (4-way ring) ─────────────────────────────
        # Technical ← Sentiment:  Tech attends to Sent
        self.cross_tech_sent = CrossModalAttention(
            d_model=D,
            n_heads=config.n_heads,
            dropout=config.attention_dropout,
        )
        # Sentiment ← Fundamental: Sent attends to Fund
        self.cross_sent_fund = CrossModalAttention(
            d_model=D,
            n_heads=config.n_heads,
            dropout=config.attention_dropout,
        )
        # Fundamental ← Macro:      Fund attends to Macro
        self.cross_fund_macro = CrossModalAttention(
            d_model=D,
            n_heads=config.n_heads,
            dropout=config.attention_dropout,
        )
        # Macro ← Technical:        Macro attends to Tech
        self.cross_macro_tech = CrossModalAttention(
            d_model=D,
            n_heads=config.n_heads,
            dropout=config.attention_dropout,
        )

        # ── 4. Dynamic Gating Network (macro-only → 3-way) ──────────────────
        # KWARG FIX (Audit Fix NEW-C2): n_modalities → n_gated_modalities,
        # value 4 → 3 (tech/fund/sent gated by macro context).
        self.gating = DynamicGatingNetwork(
            d_model=D,
            n_gated_modalities=3,
            hidden_dim=max(D // 2, 64),
            dropout=config.dropout,
        )

        # ── 5. Residual Fusion Block ──────────────────────────────────────────
        self.fusion = ResidualFusionBlock(
            d_model=D,
            fusion_dim=config.fusion_dim,
            dropout=config.dropout,
        )

        # ── 6. Global Transformer Encoder Blocks ─────────────────────────────
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=D,
                n_heads=config.n_heads,
                ff_mult=4,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
            )
            for _ in range(config.n_transformer_blocks)
        ])

        self.pre_head_norm = nn.LayerNorm(D, eps=1e-6)

        # ── 7. Triple Prediction Heads ────────────────────────────────────────
        # Each head corresponds to one trading frequency:
        #   head_1d   — daily rebalancing  — driven by gate_tech  (fast signal)
        #   head_21d  — monthly rebalancing — driven by gate_macro (medium signal)
        #   head_126d — 6-month rebalancing — driven by gate_fund  (slow signal)
        head_hidden = max(D // 2, 64)
        self.head_1d   = PredictionHead(D, head_hidden, dropout=config.dropout)
        self.head_21d  = PredictionHead(D, head_hidden, dropout=config.dropout)
        self.head_126d = PredictionHead(D, head_hidden, dropout=config.dropout)

        # Weight initialisation
        self._init_weights()

    # ────────────────────────────────────────────────────────────────────────
    def _init_weights(self) -> None:
        """Apply Xavier uniform initialisation to all Linear layers.

        Biases are initialised to zero; LayerNorm weights to 1 and biases to 0.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        x_tech: torch.Tensor,
        x_fund: torch.Tensor,
        x_macro: torch.Tensor,
        x_sent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
        """Full forward pass.

        Parameters
        ----------
        x_tech : torch.Tensor
            Technical features ``(B, tech_dim)``.  Fast, daily signal.
        x_fund : torch.Tensor
            Fundamental features ``(B, fund_dim)``.  Slow, quarterly signal.
        x_macro : torch.Tensor
            Macro features ``(B, macro_dim)``.  Medium-frequency signal.
        x_sent : torch.Tensor
            Sentiment features ``(B, sent_dim)``.

        Returns
        -------
        pred_1d : torch.Tensor
            Predicted 1-day return ``(B, 1)``.
        pred_21d : torch.Tensor
            Predicted 21-day return ``(B, 1)``.
        pred_126d : torch.Tensor
            Predicted 126-day (6-month) return ``(B, 1)``.
        aux : Dict
            Interpretability dictionary (gates, attention weights).
            Includes ``gate_tech``, ``gate_fund``, ``gate_sent``.
        """
        # ── Step 1: Independent modal encoding ───────────────────────────────
        # Outputs: (B, T, D)
        h_tech  = self.tech_encoder(x_tech)    # (B, T, D)
        h_fund  = self.fund_encoder(x_fund)    # (B, T, D)
        h_macro = self.macro_encoder(x_macro)  # (B, T, D)
        h_sent  = self.sent_encoder(x_sent)    # (B, T, D)

        # ── Step 2: Per-modal multi-head self-attention ───────────────────────
        h_tech,  aw_tech  = self.mhsa_tech(h_tech)    # (B, T, D), (B, T, T)
        h_fund,  aw_fund  = self.mhsa_fund(h_fund)
        h_macro, aw_macro = self.mhsa_macro(h_macro)
        h_sent,  aw_sent  = self.mhsa_sent(h_sent)

        # ── Step 3: Sequential cross-modal attention (4-way ring) ──────────────
        # SEQUENTIAL RING ATTENTION FIX (Audit Fix M2)
        # Each step uses the UPDATED output from the previous step,
        # implementing true cyclic information flow as described in the thesis:
        #   Tech ← Sent(updated) ← Fund(updated) ← Macro(updated) ← Tech
        
        # Step 3a: Fund (Q) ← Macro (K,V) — macro context informs fundamentals
        h_fund_cross,  cw_fm = self.cross_fund_macro(h_fund, h_macro)
        # Step 3b: Sent (Q) ← Fund_updated (K,V) — updated fundamentals inform sentiment
        h_sent_cross,  cw_sf = self.cross_sent_fund(h_sent, h_fund_cross)
        # Step 3c: Tech (Q) ← Sent_updated (K,V) — updated sentiment informs technicals
        h_tech_cross,  cw_ts = self.cross_tech_sent(h_tech, h_sent_cross)
        # Step 3d: Macro (Q) ← Tech_updated (K,V) — updated technicals modulate macro
        h_macro_cross, cw_mt = self.cross_macro_tech(h_macro, h_tech_cross)

        # ── Step 4: Dynamic gating (macro-only conditioned, 3-way) ─────────────
        gates = self.gating(h_tech_cross, h_fund_cross, h_macro_cross, h_sent_cross)   # (B, 3)

        # ── Step 5: Residual fusion (tech + fund + sent; macro excluded) ──────
        h_fused = self.fusion(h_tech_cross, h_fund_cross, h_macro_cross, h_sent_cross, gates)  # (B, T, D)

        # ── Step 6: Global transformer blocks ────────────────────────────────
        h = h_fused
        tx_aw: List[torch.Tensor] = []
        for block in self.transformer_blocks:
            h, block_aw = block(h)
            tx_aw.append(block_aw)

        # Pool the last step of the sequence (h[:, -1, :]) for the prediction heads
        h_pooled = h[:, -1, :]  # (B, D)
        h_pooled = self.pre_head_norm(h_pooled)

        # ── Step 7: Triple prediction heads ──────────────────────────────────
        pred_1d   = self.head_1d(h_pooled)    # (B, 1)
        pred_21d  = self.head_21d(h_pooled)   # (B, 1)
        pred_126d = self.head_126d(h_pooled)  # (B, 1)

        # AUX DICTIONARY FIX (Audit Fix NEW-C3)
        # =========================================
        # After C7: gates shape is (B, 3) = [w_tech, w_fund, w_sent]
        # Macro is the conditioning signal, not a gated modality.
        aux = {
            "gates":          gates,                   # (B, 3) [tech, fund, sent]
            "gate_tech":      gates[:, 0:1],           # (B, 1) technical gate weight
            "gate_fund":      gates[:, 1:2],           # (B, 1) fundamental gate weight
            "gate_sent":      gates[:, 2:3],           # (B, 1) sentiment gate weight
            "attn_tech":      aw_tech,
            "attn_fund":      aw_fund,
            "attn_macro":     aw_macro,
            "attn_sent":      aw_sent,
            "cross_ts":       cw_ts,
            "cross_sf":       cw_sf,
            "cross_fm":       cw_fm,
            "cross_mt":       cw_mt,
            "transformer_aw": tx_aw,
        }

        return pred_1d, pred_21d, pred_126d, aux

    # ────────────────────────────────────────────────────────────────────────
    def get_gate_weights(
        self,
        x_tech: torch.Tensor,
        x_fund: torch.Tensor,
        x_macro: torch.Tensor,
        x_sent: torch.Tensor,
    ) -> torch.Tensor:
        """Return only the dynamic gate weights (no gradients).

        Convenience method for post-hoc interpretability analysis.

        Parameters
        ----------
        x_tech, x_fund, x_macro, x_sent : torch.Tensor
            Input feature tensors (same shapes as :meth:`forward`).

        Returns
        -------
        torch.Tensor
            Gate weights ``(B, 4)`` with columns [tech, fund, macro, sent].
        """
        with torch.no_grad():
            _, _, _, aux = self.forward(x_tech, x_fund, x_macro, x_sent)
        return aux["gates"]

    # ────────────────────────────────────────────────────────────────────────
    def extra_repr(self) -> str:
        cfg = self.config
        return (
            f"tech_dim={cfg.tech_dim}, fund_dim={cfg.fund_dim}, macro_dim={cfg.macro_dim}, "
            f"d_model={cfg.d_model}, n_heads={cfg.n_heads}, "
            f"n_encoder_layers={cfg.n_encoder_layers}, "
            f"n_transformer_blocks={cfg.n_transformer_blocks}, "
            f"fusion_dim={cfg.fusion_dim}, dropout={cfg.dropout}"
        )


# ─── Factory ─────────────────────────────────────────────────────────────────

def build_model(config: TFDMGAConfig) -> TFDMGA:
    """Instantiate a TFDMGA model from a configuration object.

    Parameters
    ----------
    config : TFDMGAConfig
        Master configuration.

    Returns
    -------
    TFDMGA
        Initialised model (not yet compiled or moved to device).
    """
    return TFDMGA(config)


# ─── Self-contained smoke test ────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    from .utils import format_bytes, get_gpu_memory_info, set_seed

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning full model smoke test on: {device}")

    cfg = TFDMGAConfig(
        tech_dim=46,
        fund_dim=192,
        macro_dim=26,
        sent_dim=2,
        d_model=256,
        n_heads=8,
        n_encoder_layers=3,
        n_transformer_blocks=4,
        fusion_dim=512,
        dropout=0.1,
        use_compile=False,   # disable compile for smoke test
    )

    model = build_model(cfg).to(device)
    model.train()

    logger = setup_logger("smoke_test", "/tmp/tfdmga_logs")
    log_model_summary(model, logger)

    B = 32
    T = cfg.window_size
    x_tech  = torch.randn(B, T, cfg.tech_dim,  device=device)
    x_fund  = torch.randn(B, T, cfg.fund_dim,  device=device)
    x_macro = torch.randn(B, T, cfg.macro_dim, device=device)
    x_sent  = torch.randn(B, T, cfg.sent_dim,  device=device)

    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        pred_1d, pred_21d, pred_126d, aux = model(x_tech, x_fund, x_macro, x_sent)

    print(f"  pred_1d  : {tuple(pred_1d.shape)}")
    print(f"  pred_21d : {tuple(pred_21d.shape)}")
    print(f"  pred_126d: {tuple(pred_126d.shape)}")
    print(f"  gates    : {tuple(aux['gates'].shape)}")
    print(f"  gate mean: {aux['gates'].mean(0).detach().cpu().numpy()}")
    assert pred_1d.shape  == (B, 1), f"Shape error: {pred_1d.shape}"
    assert pred_21d.shape == (B, 1), f"Shape error: {pred_21d.shape}"
    assert pred_126d.shape == (B, 1), f"Shape error: {pred_126d.shape}"
    assert aux["gates"].shape == (B, 4)

    # Gradient flow check
    loss = pred_1d.mean() + pred_21d.mean() + pred_126d.mean()
    loss.backward()
    grads_ok = all(
        p.grad is not None for p in model.parameters() if p.requires_grad
    )
    print(f"  Gradient flow: {'OK' if grads_ok else 'FAILED'}")

    n_params = count_parameters(model)
    print(f"  Trainable parameters: {n_params:,}")

    info = get_gpu_memory_info(device)
    print(f"  GPU memory used: {format_bytes(info['used'])}")

    print("\nFull TFDMGA model smoke test PASSED.")
