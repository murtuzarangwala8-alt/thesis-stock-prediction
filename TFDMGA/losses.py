"""
losses.py — TFDMGA Multi-Task Loss Functions
=============================================
Implements all loss components used during training:

  * HuberLoss           — robust regression loss
  * MSELoss             — standard mean squared error
  * RankingLoss         — pairwise ranking loss (ListMLE-style)
  * ICLoss              — maximises Pearson correlation (Information Coefficient)
  * MultiTaskLoss       — weighted combination of the above for both targets

Author: TFDMGA Research Framework
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


# ─── Huber Loss ──────────────────────────────────────────────────────────────

class HuberLoss(nn.Module):
    """Huber (smooth L1) loss with configurable delta.

    For |y - ŷ| ≤ delta: L = 0.5 · (y - ŷ)²
    For |y - ŷ| > delta: L = delta · (|y - ŷ| - 0.5 · delta)

    This is more robust to outliers than MSE while maintaining quadratic
    behaviour for small errors.

    Parameters
    ----------
    delta : float
        Transition point between L2 and L1 behaviour.
    reduction : str
        ``"mean"`` or ``"sum"``.
    """

    def __init__(self, delta: float = 0.5, reduction: str = "mean") -> None:
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        self._loss = nn.HuberLoss(delta=delta, reduction=reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Huber loss.

        Parameters
        ----------
        pred : torch.Tensor
            Model predictions ``(B, 1)`` or ``(B,)``.
        target : torch.Tensor
            Ground-truth returns ``(B, 1)`` or ``(B,)``.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        return self._loss(pred.view(-1), target.view(-1))


# ─── MSE Loss ────────────────────────────────────────────────────────────────

class MSELoss(nn.Module):
    """Standard mean squared error loss.

    Parameters
    ----------
    reduction : str
        ``"mean"`` or ``"sum"``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self._loss = nn.MSELoss(reduction=reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss(pred.view(-1), target.view(-1))


# ─── Ranking Loss (ListMLE-style pairwise) ────────────────────────────────────

class RankingLoss(nn.Module):
    """Pairwise ranking loss that encourages correct ordering of predictions.

    For each pair (i, j) where target_i > target_j, the model is penalised
    if pred_i < pred_j. Implemented as a soft hinge:

        L = mean_{i,j: y_i > y_j} max(0, margin - (pred_i - pred_j))

    This is equivalent to a pairwise SVM ranking loss and is known to improve
    rank-IC when added to regression objectives.

    Parameters
    ----------
    margin : float
        Minimum required score separation between correctly-ranked pairs.
    sample_pairs : int
        Number of random pairs sampled per batch to keep memory tractable.
        Full O(B²) computation is used when ``sample_pairs <= 0``.
    """

    def __init__(self, margin: float = 0.0, sample_pairs: int = 2048) -> None:
        super().__init__()
        self.margin = margin
        self.sample_pairs = sample_pairs

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute pairwise ranking loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted scores ``(B,)``.
        target : torch.Tensor
            True return values ``(B,)``.

        Returns
        -------
        torch.Tensor
            Scalar mean pairwise hinge loss.
        """
        pred   = pred.view(-1)
        target = target.view(-1)
        B = pred.size(0)

        if self.sample_pairs > 0 and B * (B - 1) // 2 > self.sample_pairs:
            # Sample random pairs to keep memory O(sample_pairs) instead of O(B²)
            idx_i = torch.randint(0, B, (self.sample_pairs,), device=pred.device)
            idx_j = torch.randint(0, B, (self.sample_pairs,), device=pred.device)
            # Remove self-pairs
            mask = idx_i != idx_j
            idx_i, idx_j = idx_i[mask], idx_j[mask]
        else:
            # All pairs (bilateral: include both (i, j) and (j, i))
            idx_i_tri, idx_j_tri = torch.triu_indices(B, B, offset=1, device=pred.device)
            idx_i = torch.cat([idx_i_tri, idx_j_tri])
            idx_j = torch.cat([idx_j_tri, idx_i_tri])

        diff_target = target[idx_i] - target[idx_j]
        diff_pred   = pred[idx_i]   - pred[idx_j]

        # Only penalise pairs where target_i > target_j but pred_i <= pred_j
        should_rank_higher = diff_target > 0
        loss_pairs = F.relu(self.margin - diff_pred[should_rank_higher])

        if loss_pairs.numel() == 0:
            return pred.new_zeros(())
        return loss_pairs.mean()


# ─── IC Loss (Pearson Correlation Maximisation) ───────────────────────────────

class ICLoss(nn.Module):
    """Differentiable Information Coefficient (IC) loss.

    Minimises the negative Pearson correlation between predictions and targets.
    Directly optimising IC is the most theoretically grounded objective for
    financial return prediction, as IC measures the cross-sectional signal
    quality of the predictor.

        L_IC = 1 - Pearson(pred, target)

    Parameters
    ----------
    eps : float
        Small constant for numerical stability in the standard deviation.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute negative Pearson IC loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted scores ``(B,)`` or ``(B, 1)``.
        target : torch.Tensor
            True returns ``(B,)`` or ``(B, 1)``.

        Returns
        -------
        torch.Tensor
            Scalar loss equal to (1 − IC), where IC ∈ [−1, 1].
        """
        pred   = pred.view(-1)
        target = target.view(-1)

        pred_mean   = pred.mean()
        target_mean = target.mean()

        pred_dev   = pred   - pred_mean
        target_dev = target - target_mean

        cov = (pred_dev * target_dev).mean()
        std_pred   = pred_dev.pow(2).mean().sqrt().clamp(min=1e-5)
        std_target = target_dev.pow(2).mean().sqrt().clamp(min=1e-5)

        ic = cov / (std_pred * std_target + 1e-7)
        ic = torch.nan_to_num(ic, nan=0.0, posinf=1.0, neginf=-1.0)
        return 1.0 - ic   # minimise → maximise IC


# ─── Multi-Task Loss ──────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """Weighted multi-task loss for 1-day, 21-day, and 126-day prediction.

    Combines the primary regression loss with optional ranking and IC components:

        L = w_1d   · L_reg(pred_1d,   y_1d)
          + w_21d  · L_reg(pred_21d,  y_21d)
          + w_126d · L_reg(pred_126d, y_126d)       [if include_126d=True]
          + w_rank · L_rank(pred_1d,  y_1d)          [if enabled]
          + w_ic   · L_ic(pred_1d,    y_1d)           [if enabled]

    The three regression targets correspond to three trading frequencies:
      * 1d   — daily rebalancing    (technical signal horizon)
      * 21d  — monthly rebalancing  (macro signal horizon)
      * 126d — 6-month rebalancing  (fundamental signal horizon)

    Parameters
    ----------
    loss_type : str
        ``"huber"``, ``"mse"``, or ``"hybrid"`` (average of both).
    huber_delta : float
        Delta for the Huber loss.
    loss_weight_1d : float
        Weight for the 1-day regression loss component.
    loss_weight_21d : float
        Weight for the 21-day regression loss component.
    loss_weight_126d : float
        Weight for the 126-day regression loss component.
    include_126d : bool
        If False, the 126-day head loss is ignored (weights must still sum to 1.0
        across 1d and 21d in that case).
    use_ranking_loss : bool
        Add the pairwise ranking loss.
    ranking_loss_weight : float
        Weight of the ranking loss in the combined objective.
    use_ic_loss : bool
        Add the IC-maximisation loss.
    ic_loss_weight : float
        Weight of the IC loss.
    """

    def __init__(
        self,
        loss_type: str = "huber",
        huber_delta: float = 0.5,
        loss_weight_1d: float   = 0.40,
        loss_weight_21d: float  = 0.35,
        loss_weight_126d: float = 0.25,
        include_126d: bool = True,
        use_ranking_loss: bool = True,
        ranking_loss_weight: float = 0.10,
        use_ic_loss: bool = False,
        ic_loss_weight: float = 0.05,
    ) -> None:
        super().__init__()

        self.loss_weight_1d   = loss_weight_1d
        self.loss_weight_21d  = loss_weight_21d
        self.loss_weight_126d = loss_weight_126d
        self.include_126d     = include_126d
        self.use_ranking_loss    = use_ranking_loss
        self.ranking_loss_weight = ranking_loss_weight
        self.use_ic_loss    = use_ic_loss
        self.ic_loss_weight = ic_loss_weight

        # Primary regression losses (one per horizon)
        def _make_reg(lt: str) -> nn.Module:
            """Create a regression loss module for one head."""
            if lt == "huber":  return HuberLoss(delta=huber_delta)
            if lt == "mse":    return MSELoss()
            raise ValueError(f"Unknown loss_type: {lt}")

        if loss_type == "hybrid":
            # Hybrid: arithmetic mean of Huber and MSE
            self.huber_1d   = HuberLoss(delta=huber_delta)
            self.mse_1d     = MSELoss()
            self.huber_21d  = HuberLoss(delta=huber_delta)
            self.mse_21d    = MSELoss()
            self.huber_126d = HuberLoss(delta=huber_delta)
            self.mse_126d   = MSELoss()
            self.reg_loss_1d   = None
            self.reg_loss_21d  = None
            self.reg_loss_126d = None
        elif loss_type in ("huber", "mse"):
            self.reg_loss_1d   = _make_reg(loss_type)
            self.reg_loss_21d  = _make_reg(loss_type)
            self.reg_loss_126d = _make_reg(loss_type)
        else:
            raise ValueError(f"Unknown loss_type: '{loss_type}'. Choose huber/mse/hybrid.")

        self.loss_type = loss_type

        # Optional auxiliary losses
        if use_ranking_loss:
            self.ranking_loss = RankingLoss(margin=0.0, sample_pairs=2048)
        if use_ic_loss:
            self.ic_loss = ICLoss()

    def _compute_reg(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        horizon: str,      # "1d", "21d", or "126d"
    ) -> torch.Tensor:
        """Internal helper to compute the primary regression loss for one horizon."""
        if self.loss_type == "hybrid":
            huber = getattr(self, f"huber_{horizon}")
            mse   = getattr(self, f"mse_{horizon}")
            return 0.5 * (huber(pred, target) + mse(pred, target))
        reg = getattr(self, f"reg_loss_{horizon}")
        return reg(pred, target)

    def forward(
        self,
        pred_1d:   torch.Tensor,
        pred_21d:  torch.Tensor,
        pred_126d: torch.Tensor,
        y_1d:      torch.Tensor,
        y_21d:     torch.Tensor,
        y_126d:    torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the combined multi-task loss across all three horizons.

        Parameters
        ----------
        pred_1d   : torch.Tensor  (B, 1) or (B,)  — daily prediction
        pred_21d  : torch.Tensor  (B, 1) or (B,)  — monthly prediction
        pred_126d : torch.Tensor  (B, 1) or (B,)  — 6-month prediction
        y_1d      : torch.Tensor  (B, 1) or (B,)
        y_21d     : torch.Tensor  (B, 1) or (B,)
        y_126d    : torch.Tensor  (B, 1) or (B,)

        Returns
        -------
        total_loss : torch.Tensor
            Scalar combined loss used for ``loss.backward()``.
        components : Dict[str, torch.Tensor]
            Named breakdown of each loss component for logging.
        """
        reg_1d   = self._compute_reg(pred_1d,   y_1d,   horizon="1d")
        reg_21d  = self._compute_reg(pred_21d,  y_21d,  horizon="21d")

        total = (
            self.loss_weight_1d  * reg_1d
            + self.loss_weight_21d * reg_21d
        )

        components: Dict[str, torch.Tensor] = {
            "reg_1d":   reg_1d.detach(),
            "reg_21d":  reg_21d.detach(),
        }

        if self.include_126d:
            reg_126d = self._compute_reg(pred_126d, y_126d, horizon="126d")
            total = total + self.loss_weight_126d * reg_126d
            components["reg_126d"] = reg_126d.detach()

        if self.use_ranking_loss:
            rank_loss = self.ranking_loss(pred_1d, y_1d)
            total = total + self.ranking_loss_weight * rank_loss
            components["ranking"] = rank_loss.detach()

        if self.use_ic_loss:
            ic_l = self.ic_loss(pred_1d, y_1d)
            total = total + self.ic_loss_weight * ic_l
            components["ic"] = ic_l.detach()

        components["total"] = total.detach()
        return total, components


# ─── Self-contained smoke test ────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running losses smoke test on: {device}")

    B = 128
    pred_1d   = torch.randn(B, 1, device=device, requires_grad=True)
    pred_21d  = torch.randn(B, 1, device=device, requires_grad=True)
    pred_126d = torch.randn(B, 1, device=device, requires_grad=True)
    y_1d      = torch.randn(B, 1, device=device)
    y_21d     = torch.randn(B, 1, device=device)
    y_126d    = torch.randn(B, 1, device=device)

    for loss_type in ("huber", "mse", "hybrid"):
        mtl = MultiTaskLoss(
            loss_type=loss_type,
            loss_weight_1d=0.40, loss_weight_21d=0.35, loss_weight_126d=0.25,
            include_126d=True,
            use_ranking_loss=True,
            use_ic_loss=True,
        ).to(device)
        total, comps = mtl(pred_1d, pred_21d, pred_126d, y_1d, y_21d, y_126d)
        total.backward(retain_graph=True)
        print(f"  loss_type={loss_type:6s}: total={total.item():.4f} | "
              f"1d={comps['reg_1d'].item():.4f} | "
              f"21d={comps['reg_21d'].item():.4f} | "
              f"126d={comps['reg_126d'].item():.4f} | "
              f"rank={comps.get('ranking', torch.tensor(0)).item():.4f}")

    print("All loss tests passed.")
