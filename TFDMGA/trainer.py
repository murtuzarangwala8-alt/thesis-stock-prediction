"""
trainer.py — TFDMGA Training Engine
=====================================
Production training loop implementing every GPU performance optimisation:

  * torch.compile (reduce-overhead mode)
  * Automatic Mixed Precision (AMP) with GradScaler
  * AdamW fused optimiser (CUDA-fused kernel)
  * Cosine annealing LR scheduler with linear warmup
  * Gradient clipping
  * TF32 matrix math
  * Non-blocking GPU tensor transfers
  * TensorBoard + CSV logging
  * GPU memory logging per epoch
  * Checkpoint save / resume
  * Best model auto-save (by val IC-IR)
  * Early stopping on val IC-IR

Author: TFDMGA Research Framework
"""
from __future__ import annotations

import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

from .config import TFDMGAConfig
from .dataset import FinancialPanelDataset, make_dataloader
from .losses import MultiTaskLoss
from .metrics import compute_daily_ic, compute_icir, evaluate_predictions
from .model import TFDMGA, build_model
from .utils import (
    CSVLogger,
    Timer,
    count_parameters,
    format_bytes,
    get_gpu_memory_info,
    load_checkpoint,
    log_model_summary,
    save_checkpoint,
    set_seed,
    setup_logger,
    warmup_gpu,
)


# ─── Learning Rate Schedule ───────────────────────────────────────────────────

def _build_warmup_cosine_scheduler(
    optimiser: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Construct a linear warmup → cosine annealing LR schedule.

    Parameters
    ----------
    optimiser : torch.optim.Optimizer
    warmup_steps : int
        Number of steps for linear warmup (from 0 to peak LR).
    total_steps : int
        Total number of training steps (warmup + cosine decay).

    Returns
    -------
    LambdaLR
        Scheduler instance.
    """
    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(warmup_steps, 1)
        progress = float(step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimiser, lr_lambda=_lr_lambda)


# ─── EarlyStopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when the monitored metric fails to improve.

    The model's best weights are stored in ``self.best_state_dict`` so they
    can be restored after stopping.

    Parameters
    ----------
    patience : int
        Number of non-improving epochs before stopping.
    min_delta : float
        Minimum improvement required to reset the patience counter.
    mode : str
        ``"max"`` if higher is better (e.g. IC-IR), ``"min"`` otherwise.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-5,
        mode: str = "max",
    ) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.counter:       int = 0
        self.best_value: Optional[float] = None
        self.best_state_dict: Optional[Dict] = None
        self.should_stop:   bool = False

    def __call__(self, value: float, model: nn.Module) -> bool:
        """Update state with a new metric value.

        Parameters
        ----------
        value : float
            Current epoch metric value.
        model : nn.Module
            Model whose weights are saved when a new best is found.

        Returns
        -------
        bool
            ``True`` if training should stop.
        """
        improved = (
            self.best_value is None
            or (self.mode == "max" and value > self.best_value + self.min_delta)
            or (self.mode == "min" and value < self.best_value - self.min_delta)
        )
        if improved:
            self.best_value = value
            self.best_state_dict = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ─── Trainer ─────────────────────────────────────────────────────────────────

class Trainer:
    """Full production training engine for TFDMGA.

    Parameters
    ----------
    config : TFDMGAConfig
    fold_idx : int
        Current fold index (1-indexed). Used for checkpoint naming.
    device : torch.device
    logger : Optional[logging.Logger]
    """

    def __init__(
        self,
        config: TFDMGAConfig,
        fold_idx: int,
        device: torch.device,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config   = config
        self.fold_idx = fold_idx
        self.device   = device
        self.logger   = logger or setup_logger(f"Trainer_fold{fold_idx}", config.log_dir)

        # ── Hardware flags ────────────────────────────────────────────────────
        if config.use_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            self.logger.info("TF32 and cuDNN benchmark enabled.")

        # ── AMP scaler ────────────────────────────────────────────────────────
        self.use_amp = config.use_amp and device.type == "cuda"
        self.scaler  = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # ── Loss function ─────────────────────────────────────────────────────
        self.criterion = MultiTaskLoss(
            loss_type=config.loss_type,
            huber_delta=config.huber_delta,
            loss_weight_1d=config.loss_weight_1d,
            loss_weight_21d=config.loss_weight_21d,
            loss_weight_126d=config.loss_weight_126d,
            include_126d=config.include_126d_target,
            use_ranking_loss=config.use_ranking_loss,
            ranking_loss_weight=config.ranking_loss_weight,
            use_ic_loss=config.use_ic_loss,
            ic_loss_weight=config.ic_loss_weight,
        ).to(device)

        # ── CSV logger ────────────────────────────────────────────────────────
        self.csv_logger = CSVLogger(
            os.path.join(config.log_dir, f"fold{fold_idx}_metrics.csv")
        )

        # ── TensorBoard ───────────────────────────────────────────────────────
        self.writer: Optional["SummaryWriter"] = None
        if config.tensorboard and _TB_AVAILABLE:
            tb_dir = os.path.join(config.log_dir, "tensorboard", f"fold{fold_idx}")
            self.writer = SummaryWriter(log_dir=tb_dir)
            self.logger.info(f"TensorBoard writer at: {tb_dir}")
        elif config.tensorboard and not _TB_AVAILABLE:
            self.logger.warning(
                "TensorBoard requested but torch.utils.tensorboard not available. "
                "Install tensorboard: pip install tensorboard"
            )

        self.global_step: int = 0
        self.best_val_icir: float = -np.inf

    # ────────────────────────────────────────────────────────────────────────
    def _build_model(self) -> TFDMGA:
        """Construct, optionally compile, and move model to device."""
        model = build_model(self.config).to(self.device)
        log_model_summary(model, self.logger)

        if self.config.use_compile and hasattr(torch, "compile"):
            self.logger.info(
                f"Compiling model with mode='{self.config.compile_mode}'. "
                "First epoch will be slow (compilation)."
            )
            model = torch.compile(model, mode=self.config.compile_mode)  # type: ignore[assignment]

        return model

    # ────────────────────────────────────────────────────────────────────────
    def _build_optimiser(self, model: nn.Module) -> AdamW:
        """Construct fused AdamW optimiser.

        The fused kernel (PyTorch ≥ 2.0) executes all parameter updates
        in a single CUDA kernel launch, significantly reducing overhead
        for models with many small parameters.
        """
        fused_available = (
            "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
            and self.device.type == "cuda"
        )
        kwargs = dict(
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        if fused_available:
            kwargs["fused"] = True
            self.logger.info("Using fused AdamW CUDA kernel.")
        else:
            self.logger.info("Fused AdamW not available. Using standard AdamW.")

        return AdamW(model.parameters(), **kwargs)

    # ────────────────────────────────────────────────────────────────────────
    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimiser: AdamW,
        scheduler: LambdaLR,
        epoch: int,
    ) -> Dict[str, float]:
        """Run one full training epoch.

        Returns
        -------
        Dict[str, float]
            Averaged training loss components for this epoch.
        """
        model.train()
        accum: Dict[str, float] = {}
        n_batches = 0

        for batch in loader:
            x_tech, x_fund, x_macro, x_sent, y_1d, y_21d, y_126d = self._to_device(batch)

            optimiser.zero_grad(set_to_none=True)  # more efficient than zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                pred_1d, pred_21d, pred_126d, _ = model(x_tech, x_fund, x_macro, x_sent)
                loss, comps = self.criterion(pred_1d, pred_21d, pred_126d, y_1d, y_21d, y_126d)

            if torch.isnan(loss) or torch.isinf(loss):
                optimiser.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()
            # Unscale before clipping so that the clipping norm is in fp32 units
            self.scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
            self.scaler.step(optimiser)
            self.scaler.update()
            scheduler.step()

            # Accumulate loss components for epoch average
            for k, v in comps.items():
                accum[k] = accum.get(k, 0.0) + v.item()
            n_batches += 1
            self.global_step += 1

            # Step-level TensorBoard logging
            if self.writer and self.global_step % self.config.log_every_n_steps == 0:
                self.writer.add_scalar(
                    f"fold{self.fold_idx}/train/loss",
                    comps["total"].item(),
                    self.global_step,
                )
                self.writer.add_scalar(
                    f"fold{self.fold_idx}/lr",
                    scheduler.get_last_lr()[0],
                    self.global_step,
                )

        return {k: v / max(n_batches, 1) for k, v in accum.items()}

    # ────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _validate_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        val_df,
        pred_col: str = "pred_1d",
    ) -> Tuple[float, float, Dict[str, float]]:
        """Run validation, compute IC-IR, and return metrics.

        Returns
        -------
        val_loss : float
        val_icir : float
        metrics  : Dict[str, float]
        """
        model.eval()
        val_loss   = 0.0
        n_batches  = 0
        all_p1d:   List[np.ndarray] = []
        all_p21d:  List[np.ndarray] = []
        all_y1d:   List[np.ndarray] = []
        all_y21d:  List[np.ndarray] = []

        for batch in loader:
            x_tech, x_fund, x_macro, x_sent, y_1d, y_21d, y_126d = self._to_device(batch)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                pred_1d, pred_21d, pred_126d, _ = model(x_tech, x_fund, x_macro, x_sent)
                loss, _ = self.criterion(pred_1d, pred_21d, pred_126d, y_1d, y_21d, y_126d)
            val_loss += loss.item()
            n_batches += 1
            all_p1d.append(pred_1d.float().cpu().numpy().flatten())
            all_p21d.append(pred_21d.float().cpu().numpy().flatten())
            all_y1d.append(y_1d.float().cpu().numpy().flatten())
            all_y21d.append(y_21d.float().cpu().numpy().flatten())

        val_loss /= max(n_batches, 1)

        # Build prediction dataframe for metric computation
        val_df = val_df.copy()
        val_df[pred_col]         = np.concatenate(all_p1d)
        val_df["pred_21d"]       = np.concatenate(all_p21d)
        val_df["target_ret_1d"]  = np.concatenate(all_y1d)
        val_df["target_ret_21d"] = np.concatenate(all_y21d)

        # Compute IC-IR on val set (primary model selection metric)
        daily_ic = compute_daily_ic(val_df, pred_col=pred_col, ret_col="target_ret_1d")
        val_icir = compute_icir(daily_ic)

        return val_loss, val_icir, val_df

    # ────────────────────────────────────────────────────────────────────────
    def _to_device(self, batch: Tuple) -> Tuple[torch.Tensor, ...]:
        """Transfer a batch of tensors to the target device.

        Uses ``non_blocking=True`` for asynchronous host→device DMA when
        tensors reside in pinned host memory.
        """
        return tuple(
            t.to(self.device, non_blocking=True) if isinstance(t, torch.Tensor) else t
            for t in batch
        )

    # ────────────────────────────────────────────────────────────────────────
    def fit(
        self,
        train_dataset: FinancialPanelDataset,
        val_dataset:   FinancialPanelDataset,
        resume_from:   Optional[str] = None,
    ) -> Tuple[nn.Module, Dict[str, object]]:
        """Train the model on ``train_dataset`` and select the best checkpoint
        using ``val_dataset``.

        Parameters
        ----------
        train_dataset : FinancialPanelDataset
        val_dataset   : FinancialPanelDataset
        resume_from   : Optional[str]
            Path to an existing checkpoint to resume from.

        Returns
        -------
        model : nn.Module
            Best model (weights restored from best checkpoint).
        history : Dict[str, object]
            Dictionary of per-epoch metrics, final val IC-IR, and best epoch.
        """
        cfg    = self.config
        warmup_gpu(self.device)

        model     = self._build_model()
        optimiser = self._build_optimiser(model)

        # DataLoaders
        train_loader = make_dataloader(train_dataset, cfg.batch_size, shuffle=True,  config=cfg)
        val_loader   = make_dataloader(val_dataset,   cfg.batch_size, shuffle=False, config=cfg)

        steps_per_epoch = len(train_loader)
        warmup_steps    = cfg.warmup_epochs * steps_per_epoch
        total_steps     = cfg.max_epochs    * steps_per_epoch

        scheduler = _build_warmup_cosine_scheduler(optimiser, warmup_steps, total_steps)
        early_stop = EarlyStopping(patience=cfg.early_stopping_patience, mode="max")

        start_epoch = 0

        # ── Resume from checkpoint ────────────────────────────────────────────
        if resume_from and os.path.isfile(resume_from):
            ckpt = load_checkpoint(resume_from, self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimiser.load_state_dict(ckpt["optimiser_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            self.global_step = ckpt.get("global_step", 0)
            self.best_val_icir = ckpt.get("best_val_icir", -np.inf)
            self.logger.info(f"Resumed from checkpoint epoch {ckpt['epoch']}.")

        history: Dict[str, List] = {
            "train_loss": [], "val_loss": [], "val_icir": [], "lr": [], "epoch_time": []
        }
        val_df = val_dataset.df

        self.logger.info(
            f"Starting fold {self.fold_idx} training: "
            f"epochs={cfg.max_epochs}, batch={cfg.batch_size}, "
            f"steps/epoch={steps_per_epoch}, "
            f"warmup={warmup_steps} steps"
        )

        best_ckpt_path = os.path.join(
            cfg.checkpoint_dir, f"fold{self.fold_idx}_best.pt"
        )

        for epoch in range(start_epoch, cfg.max_epochs):
            with Timer() as epoch_timer:
                # ── Train ─────────────────────────────────────────────────────
                train_comps = self._train_epoch(
                    model, train_loader, optimiser, scheduler, epoch
                )
                # ── Validate ──────────────────────────────────────────────────
                val_loss, val_icir, _ = self._validate_epoch(
                    model, val_loader, val_df
                )

            lr_now = scheduler.get_last_lr()[0]

            # ── Logging ───────────────────────────────────────────────────────
            self.logger.info(
                f"Fold {self.fold_idx} | Epoch {epoch+1:03d}/{cfg.max_epochs} | "
                f"TrainLoss={train_comps['total']:.5f} | "
                f"ValLoss={val_loss:.5f} | ValICIR={val_icir:.4f} | "
                f"LR={lr_now:.2e} | "
                f"Time={epoch_timer.elapsed:.1f}s"
            )

            history["train_loss"].append(train_comps["total"])
            history["val_loss"].append(val_loss)
            history["val_icir"].append(val_icir)
            history["lr"].append(lr_now)
            history["epoch_time"].append(epoch_timer.elapsed)

            # GPU memory
            gpu_info = get_gpu_memory_info(self.device)
            mem_used_gb = gpu_info["used"] / 1e9

            # TensorBoard
            if self.writer:
                self.writer.add_scalar(f"fold{self.fold_idx}/val/loss",  val_loss,  epoch)
                self.writer.add_scalar(f"fold{self.fold_idx}/val/icir",  val_icir,  epoch)
                self.writer.add_scalar(f"fold{self.fold_idx}/gpu_mem_gb", mem_used_gb, epoch)

            # CSV
            self.csv_logger.log({
                "fold": self.fold_idx,
                "epoch": epoch + 1,
                "train_loss": train_comps["total"],
                **{f"train_{k}": v for k, v in train_comps.items() if k != "total"},
                "val_loss": val_loss,
                "val_icir": val_icir,
                "lr": lr_now,
                "gpu_mem_gb": mem_used_gb,
                "epoch_time_s": epoch_timer.elapsed,
            })

            # ── Best checkpoint ───────────────────────────────────────────────
            if val_icir > self.best_val_icir:
                self.best_val_icir = val_icir
                save_checkpoint(
                    state={
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimiser_state_dict": optimiser.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": self.scaler.state_dict(),
                        "val_icir": val_icir,
                        "val_loss": val_loss,
                        "best_val_icir": self.best_val_icir,
                        "global_step": self.global_step,
                        "config": self.config.to_dict(),
                    },
                    checkpoint_dir=cfg.checkpoint_dir,
                    filename=f"fold{self.fold_idx}_best.pt",
                )
                self.logger.info(
                    f"  ↑ New best val IC-IR={val_icir:.4f} — saved {best_ckpt_path}"
                )

            # Periodic checkpoint (every 10 epochs for resume safety)
            if (epoch + 1) % 10 == 0:
                save_checkpoint(
                    state={
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimiser_state_dict": optimiser.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": self.scaler.state_dict(),
                        "val_icir": val_icir,
                        "val_loss": val_loss,
                        "best_val_icir": self.best_val_icir,
                        "global_step": self.global_step,
                        "config": self.config.to_dict(),
                    },
                    checkpoint_dir=cfg.checkpoint_dir,
                    filename=f"fold{self.fold_idx}_epoch{epoch+1:03d}.pt",
                )

            # ── Early stopping ────────────────────────────────────────────────
            if early_stop(val_icir, model):
                self.logger.info(
                    f"Early stopping triggered at epoch {epoch+1} "
                    f"(patience={cfg.early_stopping_patience})."
                )
                break

        # Restore best weights
        if early_stop.best_state_dict is not None:
            # Load from file to get the absolute best across all epochs
            if os.path.isfile(best_ckpt_path):
                ckpt = load_checkpoint(best_ckpt_path, self.device)
                # Handle compiled model: strip _orig_mod prefix if present
                state = ckpt["model_state_dict"]
                try:
                    model.load_state_dict(state, strict=False)
                except Exception:
                    # Compiled model key prefix mismatch — strip prefix
                    new_state = {
                        k.replace("_orig_mod.", ""): v for k, v in state.items()
                    }
                    model.load_state_dict(new_state, strict=False)

        if self.writer:
            self.writer.flush()
            self.writer.close()

        history["best_val_icir"] = self.best_val_icir
        history["best_checkpoint"] = best_ckpt_path
        return model, history


if __name__ == "__main__":
    """Quick smoke test — runs 2 epochs on synthetic data."""
    import tempfile, os
    import pandas as pd

    rng = np.random.default_rng(0)
    n, n_tickers = 20_000, 200

    dates   = pd.date_range("2015-01-02", periods=n // n_tickers, freq="B").repeat(n_tickers)
    tickers = (["T%03d" % i for i in range(n_tickers)] * (n // n_tickers))[:n]

    cols = {"date": dates[:n], "ticker": tickers}
    for i in range(46):  cols[f"tech_{i:02d}"] = rng.normal(0, 1, n).astype(np.float32)
    for i in range(192): cols[f"fund_{i:03d}"] = rng.normal(0, 1, n).astype(np.float32)
    for i in range(26):  cols[f"macro_{i:02d}"] = rng.normal(0, 1, n).astype(np.float32)
    cols["target_ret_1d"]  = rng.normal(0, 0.01, n).astype(np.float32)
    cols["target_ret_21d"] = rng.normal(0, 0.05, n).astype(np.float32)

    df = pd.DataFrame(cols)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp = f.name
    df.to_parquet(tmp, index=False)

    from .config import TFDMGAConfig
    from .dataset import MasterDataStore, WalkForwardSplitter

    cfg = TFDMGAConfig(
        data_path=tmp,
        tech_dim=46, fund_dim=192, macro_dim=26,
        d_model=64, n_heads=4, n_encoder_layers=1, n_transformer_blocks=1,
        fusion_dim=128, max_epochs=2, batch_size=512,
        warmup_epochs=1, use_compile=False,
        checkpoint_dir="/tmp/tfdmga_trainer_test",
        log_dir="/tmp/tfdmga_trainer_logs",
        results_dir="/tmp/tfdmga_trainer_results",
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store  = MasterDataStore(cfg)
    splitter = WalkForwardSplitter(store, cfg)
    train_ds, val_ds = splitter.get_fold(1)

    trainer = Trainer(cfg, fold_idx=1, device=device)
    model, history = trainer.fit(train_ds, val_ds)

    print(f"Trainer smoke test PASSED. Best val IC-IR: {history['best_val_icir']:.4f}")
    os.unlink(tmp)
