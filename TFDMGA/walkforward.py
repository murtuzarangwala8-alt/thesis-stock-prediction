"""
walkforward.py — TFDMGA Walk-Forward Validation Engine
========================================================
Orchestrates the 5-fold expanding-window cross-validation:

  For each fold:
    1. Retrieve train/val DataSets from the shared MasterDataStore
    2. Instantiate a fresh Trainer
    3. Run training → returns best model checkpoint
    4. Evaluate fold on val set → compute all metrics
    5. Save fold summary

After all folds:
    6. Print/save aggregate fold summary table

Author: TFDMGA Research Framework
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .config import TFDMGAConfig, WALK_FORWARD_FOLDS
from .dataset import (
    FinancialPanelDataset,
    MasterDataStore,
    WalkForwardSplitter,
    make_dataloader,
)
from .metrics import evaluate_predictions, format_metrics_table
from .model import TFDMGA, build_model
from .trainer import Trainer
from .utils import (
    format_bytes,
    get_gpu_memory_info,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    setup_logger,
    Timer,
)


# ─── Per-Fold Evaluation ──────────────────────────────────────────────────────

@torch.no_grad()
def _generate_predictions(
    model: torch.nn.Module,
    dataset: FinancialPanelDataset,
    config: TFDMGAConfig,
    device: torch.device,
) -> pd.DataFrame:
    """Run the model over a dataset and return predictions concatenated with metadata.

    Parameters
    ----------
    model : nn.Module
        Trained (or compiled) model in eval mode.
    dataset : FinancialPanelDataset
    config : TFDMGAConfig
    device : torch.device

    Returns
    -------
    pd.DataFrame
        DataFrame containing the original metadata plus columns:
        ``pred_1d``, ``pred_21d``, ``gate_tech``, ``gate_fund``, ``gate_macro``.
    """
    model.eval()
    loader = make_dataloader(dataset, config.batch_size, shuffle=False, config=config)

    all_p1d:   List[np.ndarray] = []
    all_p21d:  List[np.ndarray] = []
    all_p126d: List[np.ndarray] = []
    all_gates: List[np.ndarray] = []

    for batch in loader:
        x_tech, x_fund, x_macro, x_sent, _, _, _ = tuple(
            t.to(device, non_blocking=True) if isinstance(t, torch.Tensor) else t
            for t in batch
        )
        with torch.cuda.amp.autocast(enabled=config.use_amp and device.type == "cuda"):
            pred_1d, pred_21d, pred_126d, aux = model(x_tech, x_fund, x_macro, x_sent)

        all_p1d.append(pred_1d.float().cpu().numpy().flatten())
        all_p21d.append(pred_21d.float().cpu().numpy().flatten())
        all_p126d.append(pred_126d.float().cpu().numpy().flatten())
        all_gates.append(aux["gates"].float().cpu().numpy())  # (B, 4)

    result_df = dataset.df.copy()
    result_df["pred_1d"]    = np.concatenate(all_p1d)
    result_df["pred_21d"]   = np.concatenate(all_p21d)
    result_df["pred_126d"]  = np.concatenate(all_p126d)
    gates_arr = np.concatenate(all_gates, axis=0)
    result_df["gate_tech"]  = gates_arr[:, 0]
    result_df["gate_fund"]  = gates_arr[:, 1]
    result_df["gate_macro"] = gates_arr[:, 2]
    result_df["gate_sent"]  = gates_arr[:, 3]
    return result_df


def _load_fold_model(
    fold_idx: int,
    config: TFDMGAConfig,
    device: torch.device,
    logger: logging.Logger,
) -> torch.nn.Module:
    """Load the best checkpoint for a given fold and return the model.

    Parameters
    ----------
    fold_idx : int
        1-indexed fold number.
    config : TFDMGAConfig
    device : torch.device
    logger : logging.Logger

    Returns
    -------
    nn.Module
        Model with best-fold weights loaded (eval mode).
    """
    ckpt_path = os.path.join(config.checkpoint_dir, f"fold{fold_idx}_best.pt")
    ckpt = load_checkpoint(ckpt_path, device)

    model = build_model(config).to(device)
    state = ckpt["model_state_dict"]
    # Handle compiled model prefix
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        cleaned = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(cleaned, strict=False)

    model.eval()
    logger.info(
        f"Loaded fold {fold_idx} model from {ckpt_path} "
        f"(val IC-IR={ckpt.get('val_icir', float('nan')):.4f})"
    )
    return model


# ─── Walk-Forward Engine ──────────────────────────────────────────────────────

class WalkForwardEngine:
    """Runs the complete 5-fold expanding-window validation.

    Parameters
    ----------
    config : TFDMGAConfig
    store : MasterDataStore
        The single loaded data store shared across all folds.
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
        self.config   = config
        self.store    = store
        self.device   = device
        self.logger   = logger or setup_logger("WalkForward", config.log_dir)
        self.splitter = WalkForwardSplitter(store, config)
        self.fold_results: List[Dict] = []

    # ────────────────────────────────────────────────────────────────────────
    def run(
        self,
        start_fold: int = 1,
        end_fold:   int = 5,
    ) -> List[Dict]:
        """Execute walk-forward training and evaluation for folds ``start_fold``…``end_fold``.

        Parameters
        ----------
        start_fold : int
            First fold to run (1-indexed, inclusive). Useful for resuming.
        end_fold : int
            Last fold to run (inclusive).

        Returns
        -------
        List[Dict]
            One metrics dictionary per fold.
        """
        cfg = self.config
        n_folds = self.splitter.n_folds()
        self.logger.info(
            f"Walk-forward: running folds {start_fold}–{end_fold} of {n_folds}."
        )

        for fold_idx in range(start_fold, end_fold + 1):
            fold_def = WALK_FORWARD_FOLDS[fold_idx - 1]
            self.logger.info(
                f"\n{'='*60}\n"
                f"FOLD {fold_idx}/{n_folds}  "
                f"Train {fold_def['train_years']}  →  Val {fold_def['val_years']}\n"
                f"{'='*60}"
            )
            set_seed(cfg.seed + fold_idx)   # different seed per fold for variance

            train_ds, val_ds = self.splitter.get_fold(fold_idx)
            self.logger.info(
                f"Train samples: {len(train_ds):,} | Val samples: {len(val_ds):,}"
            )

            # ── Train ─────────────────────────────────────────────────────────
            with Timer() as fold_timer:
                resume_path = os.path.join(
                    cfg.checkpoint_dir, f"fold{fold_idx}_epoch_latest.pt"
                )
                trainer = Trainer(cfg, fold_idx=fold_idx, device=self.device, logger=self.logger)
                model, train_history = trainer.fit(
                    train_ds, val_ds,
                    resume_from=resume_path if os.path.isfile(resume_path) else None,
                )

            self.logger.info(
                f"Fold {fold_idx} training complete in {fold_timer.elapsed:.1f}s. "
                f"Best val IC-IR: {train_history['best_val_icir']:.4f}"
            )

            # ── Evaluate on validation set ────────────────────────────────────
            best_model = _load_fold_model(fold_idx, cfg, self.device, self.logger)
            val_pred_df = _generate_predictions(best_model, val_ds, cfg, self.device)

            metrics_1d  = evaluate_predictions(
                val_pred_df, pred_col="pred_1d",  ret_col="target_ret_1d"
            )
            metrics_21d = evaluate_predictions(
                val_pred_df, pred_col="pred_21d", ret_col="target_ret_21d"
            )

            self.logger.info(format_metrics_table(metrics_1d,  f"Fold {fold_idx} Val — 1d"))
            self.logger.info(format_metrics_table(metrics_21d, f"Fold {fold_idx} Val — 21d"))

            # Gate weight statistics
            gate_mean = val_pred_df[["gate_tech", "gate_fund", "gate_macro"]].mean()
            self.logger.info(
                f"  Average gate weights: tech={gate_mean['gate_tech']:.3f}, "
                f"fund={gate_mean['gate_fund']:.3f}, macro={gate_mean['gate_macro']:.3f}"
            )

            fold_result = {
                "fold": fold_idx,
                "train_years": fold_def["train_years"],
                "val_years":   fold_def["val_years"],
                "train_time_s": fold_timer.elapsed,
                "best_val_icir": train_history["best_val_icir"],
                "n_epochs": len(train_history["train_loss"]),
                "gate_tech_mean":  float(gate_mean["gate_tech"]),
                "gate_fund_mean":  float(gate_mean["gate_fund"]),
                "gate_macro_mean": float(gate_mean["gate_macro"]),
                **{f"1d_{k}": v for k, v in metrics_1d.items()},
                **{f"21d_{k}": v for k, v in metrics_21d.items()},
            }
            self.fold_results.append(fold_result)

            # Save fold predictions for ensemble use
            pred_out_path = os.path.join(
                cfg.results_dir, f"fold{fold_idx}_val_predictions.parquet"
            )
            val_pred_df.to_parquet(pred_out_path, index=False)
            self.logger.info(f"Saved fold {fold_idx} val predictions → {pred_out_path}")

            # Save fold results JSON
            results_path = os.path.join(cfg.results_dir, f"fold{fold_idx}_results.json")
            with open(results_path, "w") as f:
                json.dump(fold_result, f, indent=2, default=str)

            # Free model from GPU before next fold
            del best_model
            torch.cuda.empty_cache()

        # ── Aggregate summary ─────────────────────────────────────────────────
        self._print_aggregate_summary()
        self._save_aggregate_summary()
        return self.fold_results

    # ────────────────────────────────────────────────────────────────────────
    def _print_aggregate_summary(self) -> None:
        """Print a formatted cross-fold summary table."""
        if not self.fold_results:
            return
        self.logger.info(f"\n{'='*70}")
        self.logger.info("WALK-FORWARD AGGREGATE SUMMARY")
        self.logger.info(f"{'='*70}")
        key_metrics = ["1d_ic", "1d_rank_ic", "1d_icir", "1d_sharpe", "1d_ann_ret", "best_val_icir"]
        header = f"{'Fold':<6} " + " ".join(f"{k:<14}" for k in key_metrics)
        self.logger.info(header)
        self.logger.info("-" * len(header))
        for r in self.fold_results:
            row = f"{r['fold']:<6} " + " ".join(
                f"{r.get(k, float('nan')):>14.4f}" for k in key_metrics
            )
            self.logger.info(row)
        self.logger.info("-" * len(header))
        means = {k: np.mean([r.get(k, np.nan) for r in self.fold_results]) for k in key_metrics}
        mean_row = f"{'Mean':<6} " + " ".join(f"{means[k]:>14.4f}" for k in key_metrics)
        self.logger.info(mean_row)
        self.logger.info("=" * 70)

    def _save_aggregate_summary(self) -> None:
        """Save all fold results as a single CSV and JSON."""
        if not self.fold_results:
            return
        summary_df = pd.DataFrame(self.fold_results)
        csv_path   = os.path.join(self.config.results_dir, "walkforward_summary.csv")
        json_path  = os.path.join(self.config.results_dir, "walkforward_summary.json")
        summary_df.to_csv(csv_path,   index=False)
        summary_df.to_json(json_path, orient="records", indent=2, default_handler=str)
        self.logger.info(f"Saved walk-forward summary → {csv_path}")


if __name__ == "__main__":
    print("WalkForwardEngine defined. Run via train.py for a full execution.")
