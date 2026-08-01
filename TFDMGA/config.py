"""
config.py — TFDMGA Master Configuration
========================================
Single source of truth for every hyperparameter in the framework.
All training, architecture, hardware, and data-pipeline settings live here.

Author: TFDMGA Research Framework
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Literal, Optional


@dataclass
class TFDMGAConfig:
    """
    Complete configuration for the Temporal Fusion Deep Multimodal Gated Attention Network.

    Attributes
    ----------
    tech_dim : int
        Number of technical input features (46 per thesis specification).
    fund_dim : int
        Number of fundamental input features (192 per thesis specification).
    macro_dim : int
        Number of macro input features (26 per thesis specification).
    d_model : int
        Unified latent dimension for all modal encoders and transformer blocks.
        Must be divisible by ``n_heads``.
    n_heads : int
        Number of attention heads in MHSA and cross-modal attention.
    n_encoder_layers : int
        Number of residual blocks stacked inside each modal encoder.
    n_transformer_blocks : int
        Number of global transformer encoder blocks applied after fusion.
    fusion_dim : int
        Hidden dimension of the residual fusion feed-forward network.
    dropout : float
        General dropout probability applied in encoders and fusion.
    attention_dropout : float
        Dropout probability applied to attention weight matrices.
    lr : float
        Peak learning rate for AdamW.
    weight_decay : float
        L2 regularisation coefficient for AdamW.
    batch_size : int
        Training mini-batch size.
    max_epochs : int
        Maximum number of training epochs per fold.
    grad_clip : float
        Global gradient norm clipping threshold.
    warmup_epochs : int
        Number of linear warmup epochs before cosine annealing.
    loss_weight_1d : float
        Weighting of the 1-day head loss in the combined multi-task loss.
    loss_weight_21d : float
        Weighting of the 21-day head loss in the combined multi-task loss.
    loss_type : str
        Primary regression loss. One of ``"huber"``, ``"mse"``, ``"hybrid"``.
    huber_delta : float
        Delta parameter for Huber loss (transition point from L2 to L1).
    use_ranking_loss : bool
        Whether to add a pairwise ranking loss component.
    ranking_loss_weight : float
        Weighting of the ranking loss in the combined objective.
    use_ic_loss : bool
        Whether to add an Information Coefficient (IC) maximisation loss.
    ic_loss_weight : float
        Weighting of the IC loss term.
    use_amp : bool
        Enable CUDA Automatic Mixed Precision (bfloat16/float16).
    use_compile : bool
        Enable ``torch.compile`` with the specified ``compile_mode``.
    compile_mode : str
        Mode passed to ``torch.compile``. Recommended: ``"reduce-overhead"``.
    use_tf32 : bool
        Enable TF32 for CUDA matrix multiplications (Ampere+ GPUs only).
    num_workers : int
        Number of DataLoader worker processes.
    pin_memory : bool
        Use pinned (page-locked) host memory for DataLoader.
    persistent_workers : bool
        Keep DataLoader workers alive between epochs.
    prefetch_factor : int
        Number of mini-batches pre-fetched per worker.
    data_path : str
        Absolute path to ``master_panel_features.parquet``.
    checkpoint_dir : str
        Directory for saving model checkpoints.
    log_dir : str
        Directory for TensorBoard event files and CSV logs.
    results_dir : str
        Directory for final evaluation outputs and plots.
    tech_cols : Optional[List[str]]
        Explicit list of technical feature column names. If ``None``, auto-detected.
    fund_cols : Optional[List[str]]
        Explicit list of fundamental feature column names. If ``None``, auto-detected.
    macro_cols : Optional[List[str]]
        Explicit list of macro feature column names. If ``None``, auto-detected.
    test_years : List[int]
        Years reserved exclusively for final out-of-sample testing. Never trained on.
    n_optuna_trials : int
        Total number of Optuna hyperparameter search trials.
    optuna_timeout : Optional[int]
        Wall-clock timeout in seconds for the Optuna study (``None`` = unlimited).
    seed : int
        Global random seed for full reproducibility.
    log_every_n_steps : int
        Log training metrics every this many optimiser steps.
    tensorboard : bool
        Enable TensorBoard SummaryWriter logging.
    early_stopping_patience : int
        Number of validation epochs with no IC-IR improvement before stopping.
    """

    # ── Feature dimensions (dynamically auto-detected by dataset.py) ────────
    # Default values reflect the full master panel (46 tech, 192 fund, 26 macro, 2 sent).
    # When using selected_features.json, dataset.py auto-updates these dimensions
    # to match the selected sub-panel (e.g. 20 tech, 30 fund, 7 macro, 2 sent).
    tech_dim: int = 46
    fund_dim: int = 192
    macro_dim: int = 26
    sent_dim: int = 2

    # ── Model architecture ───────────────────────────────────────────────────
    d_model: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 3
    n_transformer_blocks: int = 4
    fusion_dim: int = 512
    dropout: float = 0.10
    attention_dropout: float = 0.05
    window_size: int = 30
    tcn_channels: List[int] = field(default_factory=lambda: [128, 128])
    tcn_kernel_size: int = 3

    # ── Optimiser & scheduler ────────────────────────────────────────────────
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 2048
    max_epochs: int = 150
    grad_clip: float = 1.0
    warmup_epochs: int = 5

    # ── Multi-task loss ──────────────────────────────────────────────────────
    loss_weight_1d: float = 0.40
    loss_weight_21d: float = 0.35
    loss_weight_126d: float = 0.25        # 6-month horizon weight
    include_126d_target: bool = True      # add 6-month prediction head
    loss_type: Literal["huber", "mse", "hybrid"] = "huber"
    huber_delta: float = 0.5
    use_ranking_loss: bool = True
    ranking_loss_weight: float = 0.10
    use_ic_loss: bool = False
    ic_loss_weight: float = 0.05

    # ── Hardware / GPU optimisation flags ────────────────────────────────────
    use_amp: bool = True
    use_compile: bool = True
    compile_mode: str = "reduce-overhead"
    use_tf32: bool = True
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4

    # ── Paths ────────────────────────────────────────────────────────────────
    data_path: str = "/workspace/thesis_code/data/processed/master_panel_features.parquet"
    checkpoint_dir: str = "/workspace/checkpoints/TFDMGA"
    log_dir: str = "/workspace/logs/TFDMGA"
    results_dir: str = "/workspace/results/TFDMGA"

    # ── Column group overrides (None → auto-detect from parquet) ─────────────
    tech_cols: Optional[List[str]] = None
    fund_cols: Optional[List[str]] = None
    macro_cols: Optional[List[str]] = None
    sent_cols: Optional[List[str]] = None

    # ── Walk-forward settings ────────────────────────────────────────────────
    test_years: List[int] = field(default_factory=lambda: [2024])

    # ── Optuna ──────────────────────────────────────────────────────────────
    n_optuna_trials: int = 50
    optuna_timeout: Optional[int] = None

    # ── Reproducibility ──────────────────────────────────────────────────────
    seed: int = 42

    # ── Logging ──────────────────────────────────────────────────────────────
    log_every_n_steps: int = 50
    tensorboard: bool = True
    early_stopping_patience: int = 15

    # ────────────────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        """Validate constraints and create output directories."""
        # Dynamically load selected features if the JSON exists and we are loading the real dataset
        selected_feats_path = Path("data/processed/selected_features.json")
        if selected_feats_path.exists() and "master_panel_features" in str(self.data_path):
            try:
                import json
                with open(selected_feats_path, "r") as f:
                    sel_data = json.load(f)
                self.tech_cols = sel_data["tech_cols"]
                self.fund_cols = sel_data["fund_cols"]
                self.macro_cols = sel_data["macro_cols"]
                self.sent_cols = sel_data["sent_cols"]
                self.tech_dim = len(self.tech_cols)
                self.fund_dim = len(self.fund_cols)
                self.macro_dim = len(self.macro_cols)
                self.sent_dim = len(self.sent_cols)
            except Exception as e:
                pass

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads}). "
                f"Got remainder {self.d_model % self.n_heads}."
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1). Got {self.dropout}.")
        if not (0.0 <= self.attention_dropout < 1.0):
            raise ValueError(
                f"attention_dropout must be in [0, 1). Got {self.attention_dropout}."
            )
        total_loss = self.loss_weight_1d + self.loss_weight_21d
        if self.include_126d_target:
            total_loss += self.loss_weight_126d
        if abs(total_loss - 1.0) > 1e-4:
            raise ValueError(
                "Loss weights must sum to 1.0. "
                f"Got {total_loss:.4f} (1d={self.loss_weight_1d}, "
                f"21d={self.loss_weight_21d}, 126d={self.loss_weight_126d})."
            )
        for d in (self.checkpoint_dir, self.log_dir, self.results_dir):
            Path(d).mkdir(parents=True, exist_ok=True)

    # ────────────────────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        """Serialise the configuration to a JSON file.

        Parameters
        ----------
        path : str
            Destination file path (parent directories are created automatically).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "TFDMGAConfig":
        """Deserialise a configuration from a JSON file.

        Parameters
        ----------
        path : str
            Path to the JSON file previously written by :meth:`save`.

        Returns
        -------
        TFDMGAConfig
            Reconstructed configuration object.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Restore list fields that JSON serialises as lists already
        return cls(**data)

    def to_dict(self) -> dict:
        """Return a plain-Python-dict representation of the configuration."""
        return asdict(self)

    def __repr__(self) -> str:
        fields = "\n  ".join(
            f"{k}={v!r}" for k, v in asdict(self).items()
        )
        return f"TFDMGAConfig(\n  {fields}\n)"


# ─── Walk-forward fold definitions ──────────────────────────────────────────
# WALK-FORWARD TIMELINE FIX (Audit Fix M4)
# ==========================================
# Each fold now has explicit train/val/test years, aligned with the ML
# pipeline (src/ml_models.py). Previously, test_years was a single global
# variable [2024], meaning ALL folds shared the same test year — this is
# NOT true walk-forward evaluation.
#
# Embargo gap (C4): The walkforward.py trainer must drop the last
# embargo_days of training data before the val boundary, and the last
# embargo_days of val data before the test boundary.
WALK_FORWARD_FOLDS: List[dict] = [
    {"fold": 1, "train_years": list(range(2015, 2019)), "val_years": [2019], "test_years": [2020]},
    {"fold": 2, "train_years": list(range(2015, 2020)), "val_years": [2020], "test_years": [2021]},
    {"fold": 3, "train_years": list(range(2015, 2021)), "val_years": [2021], "test_years": [2022]},
    {"fold": 4, "train_years": list(range(2015, 2022)), "val_years": [2022], "test_years": [2023]},
    {"fold": 5, "train_years": list(range(2015, 2023)), "val_years": [2023], "test_years": [2024]},
]

# Global test years kept for backward compatibility only
TEST_YEARS: List[int] = [2020, 2021, 2022, 2023, 2024]


if __name__ == "__main__":
    cfg = TFDMGAConfig()
    print(cfg)
    tmp = "/tmp/tfdmga_config_test.json"
    cfg.save(tmp)
    cfg2 = TFDMGAConfig.load(tmp)
    assert cfg.d_model == cfg2.d_model, "Config round-trip failed."
    print("Config round-trip OK.")
    print(f"Walk-forward folds defined: {len(WALK_FORWARD_FOLDS)}")
