"""
optuna_search.py — TFDMGA Hyperparameter Optimisation
=======================================================
50-trial Optuna study over:
  - Learning rate
  - Weight decay
  - Dropout / attention dropout
  - d_model
  - n_transformer_blocks
  - n_heads
  - fusion_dim
  - n_encoder_layers
  - loss_type
  - batch_size
  - warmup_epochs
  - use_ranking_loss

Objective: mean val IC-IR on Fold 1 (fastest, most representative fold).

Uses MedianPruner to cut unpromising trials early.
Writes study results to ``results_dir/optuna_study.pkl`` and a CSV summary.

Author: TFDMGA Research Framework
"""
from __future__ import annotations

import copy
import json
import logging
import os
from typing import Dict, Optional

import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from .config import TFDMGAConfig, WALK_FORWARD_FOLDS
from .dataset import (
    FinancialPanelDataset,
    MasterDataStore,
    WalkForwardSplitter,
    make_dataloader,
)
from .losses import MultiTaskLoss
from .metrics import compute_daily_ic, compute_icir
from .model import build_model
from .utils import set_seed, setup_logger


# ─── Single trial training (fast) ────────────────────────────────────────────

def _train_trial(
    trial: optuna.Trial,
    config: TFDMGAConfig,
    train_ds: FinancialPanelDataset,
    val_ds: FinancialPanelDataset,
    device: torch.device,
    n_epochs: int = 30,
) -> float:
    """Train for ``n_epochs`` with trial-suggested hyperparameters.

    Returns the mean val IC-IR over the last 5 epochs (smoothed objective).
    """
    # ── Suggest hyperparameters ────────────────────────────────────────────────
    lr          = trial.suggest_float("lr",           1e-5, 5e-3, log=True)
    wd          = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    dropout     = trial.suggest_float("dropout",      0.0,  0.40)
    attn_drop   = trial.suggest_float("attention_dropout", 0.0, 0.20)
    d_model     = trial.suggest_categorical("d_model",     [128, 256, 512])
    n_tx_blocks = trial.suggest_int("n_transformer_blocks", 1, 6)
    n_heads     = trial.suggest_categorical("n_heads",     [4, 8])
    fusion_dim  = trial.suggest_categorical("fusion_dim",  [256, 512, 768])
    n_enc_lay   = trial.suggest_int("n_encoder_layers",    1, 4)
    loss_type   = trial.suggest_categorical("loss_type",   ["huber", "mse", "hybrid"])
    batch_size  = trial.suggest_categorical("batch_size",  [1024, 2048])
    warmup_ep   = trial.suggest_int("warmup_epochs",       1, 10)
    use_rank    = trial.suggest_categorical("use_ranking_loss", [True, False])

    # Ensure n_heads divides d_model
    while d_model % n_heads != 0:
        n_heads = n_heads // 2
        if n_heads < 1:
            n_heads = 1
            break

    # Build a trial-specific config (copy base config, override trial params)
    trial_cfg = copy.deepcopy(config)
    trial_cfg.lr                    = lr
    trial_cfg.weight_decay          = wd
    trial_cfg.dropout               = dropout
    trial_cfg.attention_dropout     = attn_drop
    trial_cfg.d_model               = d_model
    trial_cfg.n_transformer_blocks  = n_tx_blocks
    trial_cfg.n_heads               = n_heads
    trial_cfg.fusion_dim            = fusion_dim
    trial_cfg.n_encoder_layers      = n_enc_lay
    trial_cfg.loss_type             = loss_type
    trial_cfg.batch_size            = batch_size
    trial_cfg.warmup_epochs         = 1
    trial_cfg.use_ranking_loss      = use_rank
    trial_cfg.use_compile           = False   # disable compile during search (speed)
    trial_cfg.tensorboard           = False
    trial_cfg.max_epochs            = n_epochs
    trial_cfg.early_stopping_patience = 10

    # Re-validate config (d_model % n_heads)
    try:
        trial_cfg.__post_init__()
    except ValueError as e:
        raise optuna.exceptions.TrialPruned(str(e))

    use_amp = trial_cfg.use_amp and device.type == "cuda"

    model     = build_model(trial_cfg).to(device)
    criterion = MultiTaskLoss(
        loss_type=loss_type,
        loss_weight_1d=trial_cfg.loss_weight_1d,
        loss_weight_21d=trial_cfg.loss_weight_21d,
        loss_weight_126d=trial_cfg.loss_weight_126d,
        include_126d=trial_cfg.include_126d_target,
        use_ranking_loss=use_rank,
        ranking_loss_weight=0.1,
        use_ic_loss=False,
    ).to(device)

    # Fused AdamW
    fused_kw: Dict = {}
    if "fused" in torch.optim.AdamW.__init__.__code__.co_varnames and device.type == "cuda":
        fused_kw["fused"] = True
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95), **fused_kw
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_loader = make_dataloader(train_ds, batch_size, shuffle=True,  config=trial_cfg)
    val_loader   = make_dataloader(val_ds,   batch_size, shuffle=False, config=trial_cfg)

    icir_history = []

    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            x_tech, x_fund, x_macro, x_sent, y_1d, y_21d, y_126d = tuple(
                t.to(device, non_blocking=True) for t in batch
            )
            optimiser.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                p1, p2, p126, _ = model(x_tech, x_fund, x_macro, x_sent)
                loss, _   = criterion(p1, p2, p126, y_1d, y_21d, y_126d)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()

        # ── Validation IC (daily cross-sectional rank correlation) ────────────
        # OPTUNA METRIC FIX (Audit Fix NEW-M1)
        # ======================================
        # Computes Spearman Rank IC cross-sectionally per date using compute_daily_ic,
        # then takes the mean across all validation dates.
        # This accurately measures the model's daily stock-ranking ability without
        # being biased by cross-date volatility shifts or global pooling.
        model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in val_loader:
                x_tech, x_fund, x_macro, x_sent, y_1d, y_21d, y_126d = tuple(
                    t.to(device, non_blocking=True) for t in batch
                )
                with torch.cuda.amp.autocast(enabled=use_amp):
                    p1, p2, p126, _ = model(x_tech, x_fund, x_macro, x_sent)
                all_preds.append(p1.squeeze(-1).cpu())

        preds_flat = torch.cat(all_preds).numpy()
        val_df = val_ds.df.copy()
        val_df["pred_1d"] = preds_flat
        daily_ic = compute_daily_ic(val_df, pred_col="pred_1d", ret_col="target_ret_1d", method="spearman")
        val_score = float(np.mean(daily_ic)) if len(daily_ic) > 0 else 0.0

        trial.report(val_score, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_score


# ─── Optuna Study ────────────────────────────────────────────────────────────

def run_optuna_search(
    config: TFDMGAConfig,
    store: MasterDataStore,
    device: torch.device,
    n_trials: int = 50,
    n_trial_epochs: int = 30,
    logger: Optional[logging.Logger] = None,
) -> TFDMGAConfig:
    """Run the full Optuna hyperparameter search and return the best config.

    The search uses Fold 1 only (fastest) with ``n_trial_epochs`` per trial.
    A MedianPruner stops unpromising trials early to save compute.

    Parameters
    ----------
    config : TFDMGAConfig
        Base configuration. Feature dimensions must already be set.
    store : MasterDataStore
        Pre-loaded data store.
    device : torch.device
    n_trials : int
        Total number of Optuna trials.
    n_trial_epochs : int
        Number of training epochs per trial.
    logger : Optional[logging.Logger]

    Returns
    -------
    TFDMGAConfig
        Updated config with best hyperparameters applied.
    """
    log = logger or setup_logger("OptunaSearch", config.log_dir)
    set_seed(config.seed)

    splitter = WalkForwardSplitter(store, config)
    train_ds, val_ds = splitter.get_fold(1)

    log.info(
        f"Starting Optuna search: {n_trials} trials × {n_trial_epochs} epochs "
        f"on Fold 1 (train={len(train_ds):,}, val={len(val_ds):,})"
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=config.seed, n_startup_trials=10),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1),
        study_name="TFDMGA_hypersearch",
    )

    def _objective(trial: optuna.Trial) -> float:
        try:
            val = _train_trial(trial, config, train_ds, val_ds, device, n_trial_epochs)
            return val
        except Exception as e:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            log.warning(f"Trial {trial.number} pruned/skipped due to error: {e}")
            raise optuna.exceptions.TrialPruned(f"Error: {e}")
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    study.optimize(
        _objective,
        n_trials=n_trials,
        timeout=config.optuna_timeout,
        show_progress_bar=True,
        catch=(Exception,),
    )

    # ── Save study results ────────────────────────────────────────────────────
    study_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    study_csv = os.path.join(config.results_dir, "optuna_trials.csv")
    study_df.to_csv(study_csv, index=False)
    log.info(f"Optuna study saved to: {study_csv}")

    import pickle
    study_pkl = os.path.join(config.results_dir, "optuna_study.pkl")
    with open(study_pkl, "wb") as f:
        pickle.dump(study, f)
    log.info(f"Optuna study object saved to: {study_pkl}")

    # ── Extract best params ────────────────────────────────────────────────────
    best = study.best_trial
    log.info(f"\n{'='*50}")
    log.info(f"Best trial #{best.number}: val score (-loss) = {best.value:.4f}")
    log.info("Best hyperparameters:")
    for k, v in best.params.items():
        log.info(f"  {k:<30s} = {v}")
    log.info("=" * 50)

    # Save best params as JSON
    best_json = os.path.join(config.results_dir, "optuna_best_params.json")
    with open(best_json, "w") as f:
        json.dump({"best_icir": best.value, "params": best.params}, f, indent=2)

    # ── Apply best params to config ───────────────────────────────────────────
    p = best.params
    config.lr                   = p.get("lr",                   config.lr)
    config.weight_decay         = p.get("weight_decay",         config.weight_decay)
    config.dropout              = p.get("dropout",              config.dropout)
    config.attention_dropout    = p.get("attention_dropout",    config.attention_dropout)
    config.d_model              = p.get("d_model",              config.d_model)
    config.n_transformer_blocks = p.get("n_transformer_blocks", config.n_transformer_blocks)
    config.n_heads              = p.get("n_heads",              config.n_heads)
    config.fusion_dim           = p.get("fusion_dim",           config.fusion_dim)
    config.n_encoder_layers     = p.get("n_encoder_layers",     config.n_encoder_layers)
    config.loss_type            = p.get("loss_type",            config.loss_type)
    config.batch_size           = p.get("batch_size",           config.batch_size)
    config.warmup_epochs        = p.get("warmup_epochs",        config.warmup_epochs)
    config.use_ranking_loss     = p.get("use_ranking_loss",     config.use_ranking_loss)

    # Re-enable compile for actual training
    config.use_compile = True

    # Re-validate
    config.__post_init__()

    log.info("Config updated with best Optuna hyperparameters.")
    return config


if __name__ == "__main__":
    print("OptunaSearch module loaded. Call run_optuna_search() from train.py.")
