"""
train.py — TFDMGA CLI Entry Point
===================================
The single command to run on RunPod:

    python -m TFDMGA.train [OPTIONS]

Full pipeline:
  1. Parse arguments & build TFDMGAConfig
  2. Load dataset once (MasterDataStore)
  3. (Optional) Optuna hyperparameter search — 50 trials
  4. Walk-forward validation — 5 expanding folds
  5. Final ensemble evaluation on 2024 test set

Author: TFDMGA Research Framework
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

import torch

from .config import TFDMGAConfig, WALK_FORWARD_FOLDS
from .dataset import MasterDataStore
from .evaluate import Evaluator
from .optuna_search import run_optuna_search
from .utils import (
    format_bytes,
    get_gpu_memory_info,
    set_seed,
    setup_logger,
    warmup_gpu,
)
from .walkforward import WalkForwardEngine


# ─── CLI argument parser ──────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m TFDMGA.train",
        description=(
            "TFDMGA — Temporal Fusion Deep Multimodal Gated Attention Network\n"
            "Full training pipeline for the Master's Thesis."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    p.add_argument("--data_path", type=str,
                   default="/workspace/data/master_panel_features.parquet",
                   help="Path to master_panel_features.parquet")
    p.add_argument("--checkpoint_dir", type=str,
                   default="/workspace/checkpoints/TFDMGA")
    p.add_argument("--log_dir",        type=str,
                   default="/workspace/logs/TFDMGA")
    p.add_argument("--results_dir",    type=str,
                   default="/workspace/results/TFDMGA")

    # ── Architecture ──────────────────────────────────────────────────────────
    p.add_argument("--d_model",              type=int,   default=256)
    p.add_argument("--n_heads",              type=int,   default=8)
    p.add_argument("--n_encoder_layers",     type=int,   default=3)
    p.add_argument("--n_transformer_blocks", type=int,   default=4)
    p.add_argument("--fusion_dim",           type=int,   default=512)
    p.add_argument("--dropout",              type=float, default=0.10)
    p.add_argument("--attention_dropout",    type=float, default=0.05)

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--lr",              type=float, default=3e-4)
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--batch_size",      type=int,   default=2048)
    p.add_argument("--max_epochs",      type=int,   default=150)
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--warmup_epochs",   type=int,   default=5)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--loss_type",       type=str,   default="huber",
                   choices=["huber", "mse", "hybrid"])
    p.add_argument("--no_ranking_loss", action="store_true",
                   help="Disable pairwise ranking loss component.")
    p.add_argument("--use_ic_loss",     action="store_true",
                   help="Enable IC-maximisation auxiliary loss.")

    # ── Hardware ──────────────────────────────────────────────────────────────
    p.add_argument("--no_amp",     action="store_true", help="Disable AMP.")
    p.add_argument("--no_compile", action="store_true", help="Disable torch.compile.")
    p.add_argument("--no_tf32",    action="store_true", help="Disable TF32.")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device",      type=str, default="auto",
                   help="'auto', 'cuda', or 'cpu'.")

    # ── Optuna ────────────────────────────────────────────────────────────────
    p.add_argument("--run_optuna",        action="store_true",
                   help="Run Optuna hyperparameter search before walk-forward.")
    p.add_argument("--n_optuna_trials",   type=int,   default=50)
    p.add_argument("--n_trial_epochs",    type=int,   default=30,
                   help="Epochs per Optuna trial (faster = fewer).")
    p.add_argument("--optuna_timeout",    type=int,   default=None,
                   help="Max seconds for Optuna study (None = unlimited).")

    # ── Walk-forward ──────────────────────────────────────────────────────────
    p.add_argument("--start_fold", type=int, default=1,
                   help="First fold to train (for resuming).")
    p.add_argument("--end_fold",   type=int, default=5)
    p.add_argument("--skip_training",  action="store_true",
                   help="Skip training; only run ensemble evaluation.")
    p.add_argument("--skip_evaluation", action="store_true",
                   help="Skip final ensemble evaluation.")

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--config_path", type=str, default=None,
                   help="Load config from JSON (overrides all CLI flags).")
    p.add_argument("--save_config", action="store_true",
                   help="Save final config to results_dir/config.json.")
    return p


# ─── Main ────────────────────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> None:
    """Main entry point for the full TFDMGA training pipeline.

    Parameters
    ----------
    argv : Optional[list]
        Argument list (defaults to ``sys.argv[1:]`` when ``None``).
    """
    args = _build_parser().parse_args(argv)

    # ── Device selection ──────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # ── Config ────────────────────────────────────────────────────────────────
    if args.config_path and os.path.isfile(args.config_path):
        config = TFDMGAConfig.load(args.config_path)
        print(f"Loaded config from {args.config_path}")
    else:
        config = TFDMGAConfig(
            data_path       = args.data_path,
            checkpoint_dir  = args.checkpoint_dir,
            log_dir         = args.log_dir,
            results_dir     = args.results_dir,
            d_model         = args.d_model,
            n_heads         = args.n_heads,
            n_encoder_layers     = args.n_encoder_layers,
            n_transformer_blocks = args.n_transformer_blocks,
            fusion_dim      = args.fusion_dim,
            dropout         = args.dropout,
            attention_dropout= args.attention_dropout,
            lr              = args.lr,
            weight_decay    = args.weight_decay,
            batch_size      = args.batch_size,
            max_epochs      = args.max_epochs,
            grad_clip       = args.grad_clip,
            warmup_epochs   = args.warmup_epochs,
            seed            = args.seed,
            loss_type       = args.loss_type,
            use_ranking_loss= not args.no_ranking_loss,
            use_ic_loss     = args.use_ic_loss,
            use_amp         = not args.no_amp,
            use_compile     = not args.no_compile,
            use_tf32        = not args.no_tf32,
            num_workers     = args.num_workers,
            n_optuna_trials = args.n_optuna_trials,
            optuna_timeout  = args.optuna_timeout,
        )

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = setup_logger("TFDMGA_train", config.log_dir)
    logger.info(f"TFDMGA Training Pipeline — Device: {device}")

    if device.type == "cuda":
        info = get_gpu_memory_info(device)
        logger.info(
            f"GPU: {torch.cuda.get_device_name(device)} | "
            f"VRAM total={format_bytes(info['total'])} | "
            f"free={format_bytes(info['free'])}"
        )
        logger.info(f"PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")

    # ── Reproducibility ───────────────────────────────────────────────────────
    set_seed(config.seed)
    logger.info(f"Global seed set to {config.seed}.")

    # ── GPU warm-up ───────────────────────────────────────────────────────────
    warmup_gpu(device)

    # ── Load dataset ONCE ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Loading dataset (one-time load into memory)...")
    logger.info("=" * 60)
    store = MasterDataStore(config, logger=logger)
    logger.info(
        f"Dataset ready: {len(store.arr_year):,} rows, "
        f"tech={config.tech_dim}, fund={config.fund_dim}, macro={config.macro_dim}"
    )
    if store._on_gpu:
        logger.info("Dataset cached in GPU VRAM — zero-copy training mode.")
    else:
        logger.info("Dataset in pinned host memory — async streaming mode.")

    # ── Optuna ────────────────────────────────────────────────────────────────
    if args.run_optuna:
        logger.info("=" * 60)
        logger.info(f"Starting Optuna search: {config.n_optuna_trials} trials")
        logger.info("=" * 60)
        config = run_optuna_search(
            config=config,
            store=store,
            device=device,
            n_trials=config.n_optuna_trials,
            n_trial_epochs=args.n_trial_epochs,
            logger=logger,
        )
        logger.info("Optuna search complete. Config updated with best params.")
        if args.save_config:
            cfg_path = os.path.join(config.results_dir, "config_after_optuna.json")
            config.save(cfg_path)
            logger.info(f"Post-Optuna config saved → {cfg_path}")

    # ── Walk-Forward Training ─────────────────────────────────────────────────
    if not args.skip_training:
        logger.info("=" * 60)
        logger.info(
            f"Starting walk-forward training: folds {args.start_fold}–{args.end_fold}"
        )
        logger.info("=" * 60)
        engine = WalkForwardEngine(config, store, device, logger=logger)
        fold_results = engine.run(start_fold=args.start_fold, end_fold=args.end_fold)
        logger.info(
            f"Walk-forward complete. "
            f"Mean val IC-IR: "
            f"{sum(r['best_val_icir'] for r in fold_results) / len(fold_results):.4f}"
        )
        gc.collect()
        torch.cuda.empty_cache()

    # ── Final Evaluation ──────────────────────────────────────────────────────
    if not args.skip_evaluation:
        logger.info("=" * 60)
        logger.info("Running final ensemble evaluation on 2024 test set...")
        logger.info("=" * 60)
        evaluator = Evaluator(config, store, device, logger=logger)
        final_results = evaluator.run()

        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info(
            f"  Test IC          : {final_results.get('1d_ic', 'N/A'):.4f}"
        )
        logger.info(
            f"  Test Rank-IC     : {final_results.get('1d_rank_ic', 'N/A'):.4f}"
        )
        logger.info(
            f"  Test IC-IR       : {final_results.get('1d_icir', 'N/A'):.4f}"
        )
        logger.info(
            f"  Test Sharpe      : {final_results.get('1d_sharpe', 'N/A'):.4f}"
        )
        logger.info(
            f"  Test Ann Return  : {final_results.get('1d_ann_ret', 'N/A'):.4f}"
        )
        logger.info("=" * 60)

    # ── Save final config ─────────────────────────────────────────────────────
    if args.save_config or True:   # always save final config
        cfg_path = os.path.join(config.results_dir, "final_config.json")
        config.save(cfg_path)
        logger.info(f"Final config saved → {cfg_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
