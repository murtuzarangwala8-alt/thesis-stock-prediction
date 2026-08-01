"""
evaluate.py — TFDMGA Final Ensemble Evaluation
================================================
Loads all 5 fold checkpoints, runs ensemble inference on the 2024 test set,
computes the complete metric suite, and generates publication-quality plots:

  Plots produced:
    * Cumulative return curves (Long / Short / Long-Short)
    * Drawdown chart
    * Daily IC time series
    * Modality gate weight dynamics (tech / fund / macro)
    * Prediction distribution (histogram)

All results are saved to ``config.results_dir``.

Author: TFDMGA Research Framework
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch

from .config import TFDMGAConfig, TEST_YEARS, WALK_FORWARD_FOLDS
from .dataset import (
    FinancialPanelDataset,
    MasterDataStore,
    WalkForwardSplitter,
    make_dataloader,
)
from .metrics import (
    build_daily_portfolio_returns,
    compute_annual_return,
    compute_annual_volatility,
    compute_calmar,
    compute_daily_ic,
    compute_hit_ratio,
    compute_icir,
    compute_max_drawdown,
    compute_sharpe,
    compute_sortino,
    compute_net_returns_after_costs,
    compute_turnover,
    evaluate_predictions,
    format_metrics_table,
    apply_portfolio_stop_loss,
    evaluate_stop_loss_comparison,
    format_stop_loss_comparison_table,
    apply_risk_reward_stop_loss,
    apply_rr_portfolio_stop,
    evaluate_rr_stop_comparison,
    format_rr_comparison_table,
)
from .model import build_model
from .utils import format_bytes, get_gpu_memory_info, load_checkpoint, setup_logger


# ─── Ensemble inference ───────────────────────────────────────────────────────

@torch.no_grad()
def run_ensemble_inference(
    config: TFDMGAConfig,
    test_ds: FinancialPanelDataset,
    device: torch.device,
    logger: logging.Logger,
    n_folds: int = 5,
) -> pd.DataFrame:
    """Load all fold models and average their predictions on the test set.

    Parameters
    ----------
    config : TFDMGAConfig
    test_ds : FinancialPanelDataset
        The 2024 test dataset.
    device : torch.device
    logger : logging.Logger
    n_folds : int
        Number of fold models to load and average.

    Returns
    -------
    pd.DataFrame
        Test set DataFrame augmented with columns:
        ``pred_1d``, ``pred_21d``, ``gate_tech``, ``gate_fund``, ``gate_macro``
        (ensemble-averaged across all fold models).
    """
    loader  = make_dataloader(test_ds, config.batch_size, shuffle=False, config=config)
    n_test  = len(test_ds)
    use_amp = config.use_amp and device.type == "cuda"

    # Accumulators across folds
    sum_p1d    = np.zeros(n_test, dtype=np.float64)
    sum_p21d   = np.zeros(n_test, dtype=np.float64)
    sum_p126d  = np.zeros(n_test, dtype=np.float64)
    sum_gates  = np.zeros((n_test, 4), dtype=np.float64)

    loaded_folds = 0
    for fold_idx in range(1, n_folds + 1):
        ckpt_path = os.path.join(config.checkpoint_dir, f"fold{fold_idx}_best.pt")
        if not os.path.isfile(ckpt_path):
            logger.warning(f"Checkpoint not found for fold {fold_idx}: {ckpt_path}. Skipping.")
            continue

        model = build_model(config).to(device)
        ckpt  = load_checkpoint(ckpt_path, device)
        state = ckpt["model_state_dict"]
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            cleaned = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
            model.load_state_dict(cleaned, strict=False)
        model.eval()

        logger.info(
            f"  Fold {fold_idx}: val IC-IR={ckpt.get('val_icir', float('nan')):.4f}"
        )

        fold_p1d:   List[np.ndarray] = []
        fold_p21d:  List[np.ndarray] = []
        fold_p126d: List[np.ndarray] = []
        fold_gates: List[np.ndarray] = []

        for batch in loader:
            x_tech, x_fund, x_macro, x_sent, _, _, _ = tuple(
                t.to(device, non_blocking=True) if isinstance(t, torch.Tensor) else t
                for t in batch
            )
            with torch.cuda.amp.autocast(enabled=use_amp):
                p1, p2, p126, aux = model(x_tech, x_fund, x_macro, x_sent)
            fold_p1d.append(p1.float().cpu().numpy().flatten())
            fold_p21d.append(p2.float().cpu().numpy().flatten())
            fold_p126d.append(p126.float().cpu().numpy().flatten())
            fold_gates.append(aux["gates"].float().cpu().numpy())

        sum_p1d   += np.concatenate(fold_p1d)
        sum_p21d  += np.concatenate(fold_p21d)
        sum_p126d += np.concatenate(fold_p126d)
        sum_gates += np.concatenate(fold_gates, axis=0)
        loaded_folds += 1

        del model
        torch.cuda.empty_cache()

    if loaded_folds == 0:
        raise RuntimeError(
            "No fold checkpoints found. "
            f"Expected checkpoints in: {config.checkpoint_dir}"
        )

    logger.info(f"Ensemble: averaged predictions from {loaded_folds} fold models.")

    result_df = test_ds.df.copy()
    result_df["pred_1d"]    = (sum_p1d   / loaded_folds).astype(np.float32)
    result_df["pred_21d"]   = (sum_p21d  / loaded_folds).astype(np.float32)
    result_df["pred_126d"]  = (sum_p126d / loaded_folds).astype(np.float32)
    gate_mean               = sum_gates  / loaded_folds
    result_df["gate_tech"]  = gate_mean[:, 0].astype(np.float32)  # daily signal weight
    result_df["gate_fund"]  = gate_mean[:, 1].astype(np.float32)  # 6-month signal weight
    result_df["gate_macro"] = gate_mean[:, 2].astype(np.float32)  # monthly signal weight
    result_df["gate_sent"]  = gate_mean[:, 3].astype(np.float32)  # sentiment weight

    # ── Gate-weighted composite signal ───────────────────────────────────────────
    # The model's gate weights give us an economic signal about which
    # time horizon is most relevant on each day. We blend predictions
    # weighted by the appropriate gate, re-normalised so they sum to 1:
    sum_temporal = result_df["gate_tech"] + result_df["gate_macro"] + result_df["gate_fund"]
    result_df["pred_composite"] = (
        (result_df["gate_tech"]  * result_df["pred_1d"]
        + result_df["gate_macro"] * result_df["pred_21d"]
        + result_df["gate_fund"]  * result_df["pred_126d"]) / sum_temporal.clamp(min=1e-5)
    ).astype(np.float32)

    return result_df


# ─── Plot Generation ──────────────────────────────────────────────────────────

def _plot_cumulative_returns(
    port_df: pd.DataFrame,
    save_dir: str,
) -> str:
    """Plot cumulative return curves for Long, Short, and Long-Short portfolios."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for col, label, color in [
        ("ls_ret",    "Long-Short", "steelblue"),
        ("long_ret",  "Long Only",  "forestgreen"),
        ("short_ret", "Short Only", "tomato"),
    ]:
        cum = (1.0 + port_df[col]).cumprod()
        ax.plot(port_df["date"], cum, label=label, color=color, linewidth=1.5)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("TFDMGA Ensemble — Cumulative Portfolio Returns (2024)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (growth of $1)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "cumulative_returns.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_drawdown(port_df: pd.DataFrame, save_dir: str) -> str:
    """Plot the drawdown curve of the Long-Short strategy."""
    ls   = port_df["ls_ret"].values
    cum  = np.cumprod(1.0 + ls)
    peak = np.maximum.accumulate(cum)
    dd   = (cum - peak) / peak

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(port_df["date"], dd, 0, alpha=0.4, color="tomato", label="Drawdown")
    ax.plot(port_df["date"],         dd,    color="darkred", linewidth=1)
    ax.set_title("TFDMGA Ensemble — Long-Short Drawdown (2024)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "drawdown.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_daily_ic(
    result_df: pd.DataFrame,
    pred_col: str,
    ret_col: str,
    save_dir: str,
) -> str:
    """Plot rolling 21-day mean IC over the test period."""
    daily_ic = compute_daily_ic(result_df, pred_col=pred_col, ret_col=ret_col)
    dates    = sorted(result_df["date"].unique())
    if len(dates) != len(daily_ic):
        dates = dates[:len(daily_ic)]

    rolling_ic = pd.Series(daily_ic, index=dates).rolling(21, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(dates, daily_ic, width=1, alpha=0.3, color="steelblue", label="Daily IC")
    ax.plot(rolling_ic.index, rolling_ic.values, color="navy", linewidth=2, label="21d Rolling Mean IC")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"TFDMGA Ensemble — Daily IC ({pred_col} vs {ret_col})", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Information Coefficient")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f"daily_ic_{pred_col}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_gate_weights(result_df: pd.DataFrame, save_dir: str) -> str:
    """Plot time-series of mean daily modality gate weights."""
    gate_daily = (
        result_df.groupby("date")[["gate_tech", "gate_fund", "gate_macro", "gate_sent"]]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        gate_daily["date"],
        gate_daily["gate_tech"],
        gate_daily["gate_fund"],
        gate_daily["gate_macro"],
        gate_daily["gate_sent"],
        labels=["Technical", "Fundamental", "Macro", "Sentiment"],
        colors=["#e07b54", "#5b9bd5", "#70ad47", "#8e44ad"],
        alpha=0.85,
    )
    ax.set_title("TFDMGA — Dynamic Modality Gate Weights Over Time (2024)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean Gate Weight")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "gate_weights.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_prediction_distribution(
    result_df: pd.DataFrame,
    save_dir: str,
) -> str:
    """Plot histogram of 1d, 21d and 126d predictions vs actual returns."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, pred_col, ret_col, title in [
        (axes[0], "pred_1d",   "target_ret_1d",   "1-Day Horizon (Daily)"),
        (axes[1], "pred_21d",  "target_ret_21d",  "21-Day Horizon (Monthly)"),
        (axes[2], "pred_126d", "target_ret_126d", "126-Day Horizon (6-Month)"),
    ]:
        ax.hist(result_df[ret_col],  bins=80, alpha=0.6, density=True,
                color="steelblue", label="Realised Returns")
        ax.hist(result_df[pred_col], bins=80, alpha=0.6, density=True,
                color="tomato",    label="Predictions")
        ax.set_title(f"Return Distribution — {title}", fontsize=12)
        ax.set_xlabel("Return")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "prediction_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_stop_loss_comparison(
    port_df: pd.DataFrame,
    stop_threshold: float,
    reward_ratio: float,
    trailing_stop: float,
    recovery_threshold: float,
    cooldown_days: int,
    save_dir: str,
) -> str:
    """Generate and save the 3-column risk management return comparison chart."""
    from .metrics import apply_portfolio_stop_loss, apply_rr_portfolio_stop
    
    # ── Compute strategies ──
    # Gross
    gross_nav = np.cumprod(1.0 + port_df["ls_ret"].values)
    
    # Trailing Stop Only
    sl_df = apply_portfolio_stop_loss(
        port_df,
        trailing_stop=trailing_stop,
        recovery_threshold=recovery_threshold,
        cooldown_days=cooldown_days,
    )
    trail_nav = sl_df["nav_sl"].values
    
    # 2:1 R:R + Trailing
    rr_df = apply_rr_portfolio_stop(
        port_df,
        stop_threshold=stop_threshold,
        reward_ratio=reward_ratio,
        trailing_stop=trailing_stop,
        recovery_threshold=recovery_threshold,
        cooldown_days=cooldown_days,
    )
    rr_nav = rr_df["nav_full"].values
    
    # ── Plotting ──
    fig, ax = plt.subplots(figsize=(12, 6))
    dates = port_df["date"]
    
    ax.plot(dates, gross_nav, label="Gross Return (No Risk Management)", color="#D62728", linewidth=1.5, linestyle="-.")
    ax.plot(dates, trail_nav, label=f"Trailing Stop Only ({trailing_stop:.0%})", color="#FF7F0E", linewidth=1.8, linestyle="--")
    ax.plot(dates, rr_nav,    label=f"2:1 Risk-Reward ({stop_threshold:.1%}/{abs(stop_threshold)*reward_ratio:.1%}) + Trailing", color="#2CA02C", linewidth=2.5)
    
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("TFDMGA Composite Portfolio — Risk Management Overlay Performance", fontsize=14, weight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative NAV (Initial = 1.0)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(save_dir, "stop_loss_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path



# ─── Main Evaluator ───────────────────────────────────────────────────────────

class Evaluator:
    """Full test-set evaluation pipeline.

    Parameters
    ----------
    config : TFDMGAConfig
    store : MasterDataStore
    device : torch.device
    logger : Optional[logging.Logger]
    """

    def __init__(
        self,
        config: TFDMGAConfig,
        store: MasterDataStore,
        device: torch.device,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config  = config
        self.store   = store
        self.device  = device
        self.logger  = logger or setup_logger("Evaluator", config.log_dir)

    def run(self) -> Dict[str, object]:
        """Execute full evaluation on the 2024 test set.

        Returns
        -------
        Dict[str, object]
            Dictionary containing all metric values and paths to generated plots.
        """
        cfg = self.config
        splitter = WalkForwardSplitter(self.store, cfg)
        test_ds  = splitter.get_test()

        self.logger.info(
            f"\n{'='*60}\nFINAL ENSEMBLE EVALUATION — {TEST_YEARS}\n{'='*60}"
        )
        self.logger.info(f"Test samples: {len(test_ds):,}")

        # ── Ensemble inference ────────────────────────────────────────────────
        result_df = run_ensemble_inference(
            cfg, test_ds, self.device, self.logger, n_folds=len(WALK_FORWARD_FOLDS)
        )

        # Save raw predictions
        pred_path = os.path.join(cfg.results_dir, "test_predictions_ensemble.parquet")
        result_df.to_parquet(pred_path, index=False)
        self.logger.info(f"Ensemble predictions saved → {pred_path}")

        # ── Metric computation ─────────────────────────────────────────────
        metrics_1d        = evaluate_predictions(result_df, pred_col="pred_1d",        ret_col="target_ret_1d")
        metrics_21d       = evaluate_predictions(result_df, pred_col="pred_21d",       ret_col="target_ret_21d")
        metrics_126d      = evaluate_predictions(result_df, pred_col="pred_126d",      ret_col="target_ret_126d")
        metrics_composite = evaluate_predictions(result_df, pred_col="pred_composite", ret_col="target_ret_1d")

        self.logger.info(format_metrics_table(metrics_1d,        "TEST SET — 1-Day  (Daily rebalance, tech-driven)"))
        self.logger.info(format_metrics_table(metrics_21d,       "TEST SET — 21-Day (Monthly rebalance, macro-driven)"))
        self.logger.info(format_metrics_table(metrics_126d,      "TEST SET — 126-Day (6-Month rebalance, fund-driven)"))
        self.logger.info(format_metrics_table(metrics_composite, "TEST SET — Composite (gate-weighted blend)"))

        # Gate dynamics summary
        gate_mean = result_df[["gate_tech", "gate_fund", "gate_macro", "gate_sent"]].mean()
        self.logger.info(
            f"Average gate weights (test): "
            f"tech={gate_mean['gate_tech']:.3f}, "
            f"fund={gate_mean['gate_fund']:.3f}, "
            f"macro={gate_mean['gate_macro']:.3f}, "
            f"sent={gate_mean['gate_sent']:.3f}"
        )

        # ── Portfolio ─────────────────────────────────────────────────────────────
        # 1-day portfolio (daily rebalance — tech signals)
        port_df_1d  = build_daily_portfolio_returns(
            result_df, pred_col="pred_1d", ret_col="target_ret_1d"
        )
        # Composite gate-weighted portfolio
        port_df_comp = build_daily_portfolio_returns(
            result_df, pred_col="pred_composite", ret_col="target_ret_1d"
        )
        turnover   = compute_turnover(result_df, pred_col="pred_1d")
        cost_mets  = compute_net_returns_after_costs(port_df_1d, turnover)

        # ── Stop-loss on composite portfolio ──────────────────────────────────
        sl_comp = evaluate_stop_loss_comparison(
            port_df_comp, trailing_stop=-0.10, recovery_threshold=0.02, cooldown_days=5
        )
        self.logger.info(format_stop_loss_comparison_table(
            sl_comp, trailing_stop=-0.10, cooldown_days=5
        ))

        # ── 2:1 Risk-Reward full 3-column comparison table ────────────────────
        # Compares:
        #   Column 1 — Gross (no risk management)
        #   Column 2 — Trailing stop only (-10 %)
        #   Column 3 — 2:1 R:R position stops + trailing circuit breaker
        rr_comp = evaluate_rr_stop_comparison(
            port_df_comp,
            stop_threshold=-0.02,       # -2% position stop → +4% take-profit (2:1)
            reward_ratio=2.0,
            trailing_stop=-0.10,        # -10% portfolio circuit breaker
            recovery_threshold=0.02,
            cooldown_days=5,
        )
        self.logger.info(format_rr_comparison_table(
            rr_comp,
            stop_threshold=-0.02,
            reward_ratio=2.0,
            trailing_stop=-0.10,
        ))

        # ── Plot generation ───────────────────────────────────────────────────
        plot_dir = os.path.join(cfg.results_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        plots = {}
        try:
            plots["cumulative_returns"] = _plot_cumulative_returns(port_df_1d, plot_dir)
            plots["cumulative_composite"] = _plot_cumulative_returns(port_df_comp, plot_dir)
            plots["drawdown"]           = _plot_drawdown(port_df_1d, plot_dir)
            plots["ic_1d"]              = _plot_daily_ic(result_df, "pred_1d",        "target_ret_1d",   plot_dir)
            plots["ic_21d"]             = _plot_daily_ic(result_df, "pred_21d",       "target_ret_21d",  plot_dir)
            plots["ic_126d"]            = _plot_daily_ic(result_df, "pred_126d",      "target_ret_126d", plot_dir)
            plots["ic_composite"]       = _plot_daily_ic(result_df, "pred_composite", "target_ret_1d",   plot_dir)
            plots["gate_weights"]       = _plot_gate_weights(result_df, plot_dir)
            plots["distributions"]      = _plot_prediction_distribution(result_df, plot_dir)
            
            # Save risk management comparison plot
            plots["stop_loss_comparison"] = _plot_stop_loss_comparison(
                port_df_comp,
                stop_threshold=-0.02,
                reward_ratio=2.0,
                trailing_stop=-0.10,
                recovery_threshold=0.02,
                cooldown_days=5,
                save_dir=plot_dir
            )
            self.logger.info(f"Plots saved to: {plot_dir}")
        except Exception as e:
            self.logger.warning(f"Plot generation failed: {e}")

        # ── Compile full results ──────────────────────────────────────────────
        full_results = {
            "test_years":            TEST_YEARS,
            "n_test_samples":        len(result_df),
            "gate_tech_mean":        float(gate_mean["gate_tech"]),
            "gate_fund_mean":        float(gate_mean["gate_fund"]),
            "gate_macro_mean":       float(gate_mean["gate_macro"]),
            "gate_sent_mean":        float(gate_mean["gate_sent"]),
            "daily_turnover":        turnover,
            **{f"1d_{k}":        v for k, v in metrics_1d.items()},
            **{f"21d_{k}":       v for k, v in metrics_21d.items()},
            **{f"126d_{k}":      v for k, v in metrics_126d.items()},
            **{f"comp_{k}":      v for k, v in metrics_composite.items()},
            **cost_mets,
            "stop_loss_gross_sharpe":    sl_comp["gross"].get("sharpe", float("nan")),
            "stop_loss_sl_sharpe":       sl_comp["portfolio_stop"].get("sharpe", float("nan")),
            "stop_loss_flat_days":       sl_comp["portfolio_stop"].get("flat_days", 0),
            "plots": plots,
        }

        # Save to JSON
        results_path = os.path.join(cfg.results_dir, "final_evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(full_results, f, indent=2, default=str)
        self.logger.info(f"Final results saved → {results_path}")

        return full_results


if __name__ == "__main__":
    print("Evaluator module loaded. Call Evaluator(config, store, device).run() from train.py.")
