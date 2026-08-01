"""
utils.py — TFDMGA Utility Functions
=====================================
Shared infrastructure: reproducibility, GPU memory management,
structured logging, checkpoint helpers, and tensor utilities.

Author: TFDMGA Research Framework
"""
from __future__ import annotations

import csv
import gc
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ─── Logging ────────────────────────────────────────────────────────────────

def setup_logger(
    name: str,
    log_dir: str,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Create a structured logger that writes to both console and a rotating file.

    Parameters
    ----------
    name : str
        Logger name (typically the module or experiment name).
    log_dir : str
        Directory where ``{name}.log`` will be written.
    level : int
        Logging level (default ``logging.INFO``).
    console : bool
        Whether to also emit records to ``sys.stdout``.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        # Avoid duplicate handlers on re-import / re-instantiation
        logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(
        os.path.join(log_dir, f"{name}.log"), mode="a", encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    return logger


# ─── Reproducibility ────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility.

    Covers Python ``random``, NumPy, PyTorch CPU and CUDA generators,
    and enables deterministic CUDA algorithms where possible.

    Parameters
    ----------
    seed : int
        Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Use deterministic algorithms; warn rather than error if unavailable
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# ─── GPU Memory ─────────────────────────────────────────────────────────────

def get_gpu_memory_info(device: Optional[torch.device] = None) -> Dict[str, int]:
    """Return free and total VRAM in bytes for the specified CUDA device.

    Parameters
    ----------
    device : Optional[torch.device]
        Target CUDA device. Defaults to ``cuda:0``.

    Returns
    -------
    Dict[str, int]
        Dictionary with keys ``"free"``, ``"total"``, and ``"used"`` in bytes.
        Returns zeros if CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return {"free": 0, "total": 0, "used": 0}
    dev = device or torch.device("cuda:0")
    torch.cuda.synchronize(dev)
    free, total = torch.cuda.mem_get_info(dev)
    return {"free": free, "total": total, "used": total - free}


def format_bytes(n: int) -> str:
    """Human-readable byte count.

    Parameters
    ----------
    n : int
        Number of bytes.

    Returns
    -------
    str
        Formatted string, e.g. ``"3.42 GiB"``.
    """
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"


def estimate_tensor_bytes(n_rows: int, n_cols: int, dtype: torch.dtype = torch.float32) -> int:
    """Estimate the memory footprint of a 2-D tensor in bytes.

    Parameters
    ----------
    n_rows : int
        Number of rows.
    n_cols : int
        Number of columns.
    dtype : torch.dtype
        Element data type (default ``float32`` = 4 bytes/element).

    Returns
    -------
    int
        Estimated size in bytes.
    """
    bytes_per_element = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else 1
    return n_rows * n_cols * bytes_per_element


def should_cache_on_gpu(
    n_rows: int,
    n_features: int,
    n_targets: int = 2,
    safety_factor: float = 0.60,
    device: Optional[torch.device] = None,
) -> bool:
    """Decide whether the entire dataset fits in GPU VRAM with a safety margin.

    Parameters
    ----------
    n_rows : int
        Total number of data rows.
    n_features : int
        Total number of feature columns (tech + fund + macro).
    n_targets : int
        Number of target columns.
    safety_factor : float
        Fraction of free VRAM that may be used for the dataset cache.
        Default 0.60 leaves 40 % for model parameters and activations.
    device : Optional[torch.device]
        Target device.

    Returns
    -------
    bool
        ``True`` if dataset tensors fit within the VRAM safety budget.
    """
    if not torch.cuda.is_available():
        return False
    info = get_gpu_memory_info(device)
    available = int(info["free"] * safety_factor)
    needed = estimate_tensor_bytes(n_rows, n_features + n_targets)
    return needed <= available


# ─── Parameter Counting ─────────────────────────────────────────────────────

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    trainable_only : bool
        If ``True``, count only parameters with ``requires_grad=True``.

    Returns
    -------
    int
        Total parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def log_model_summary(model: nn.Module, logger: logging.Logger) -> None:
    """Log a concise model summary including parameter counts per sub-module.

    Parameters
    ----------
    model : nn.Module
        The model to summarise.
    logger : logging.Logger
        Target logger.
    """
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    logger.info("=" * 60)
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"  Total parameters   : {total:>12,}")
    logger.info(f"  Trainable params   : {trainable:>12,}")
    logger.info(f"  Non-trainable params: {total - trainable:>11,}")
    logger.info("-" * 60)
    for name, module in model.named_children():
        n = count_parameters(module, trainable_only=True)
        logger.info(f"  {name:<30s}  {n:>10,} params")
    logger.info("=" * 60)


# ─── CSV Logging ─────────────────────────────────────────────────────────────

class CSVLogger:
    """Append-mode CSV logger for tracking training / validation metrics.

    Creates the file on first write. Subsequent rows are appended without
    re-writing the header.

    Parameters
    ----------
    path : str
        File path for the CSV log.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._header_written = False
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # If file exists from a previous run, treat header as written
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            self._header_written = True

    def log(self, metrics: Dict[str, object]) -> None:
        """Append one row of metrics to the CSV file.

        Parameters
        ----------
        metrics : Dict[str, object]
            Key-value pairs to write. Keys become column headers on first call.
        """
        file_exists = os.path.isfile(self.path)
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(metrics.keys()))
            if not self._header_written or not file_exists:
                writer.writeheader()
                self._header_written = True
            writer.writerow(metrics)


# ─── Checkpoint Helpers ──────────────────────────────────────────────────────

def save_checkpoint(
    state: dict,
    checkpoint_dir: str,
    filename: str,
) -> str:
    """Save a training checkpoint dictionary to disk.

    Parameters
    ----------
    state : dict
        Dictionary containing at minimum ``model_state_dict``, ``epoch``,
        ``val_loss``, and any other desired fields.
    checkpoint_dir : str
        Directory to write the checkpoint file.
    filename : str
        Filename (without directory), e.g. ``"fold1_best.pt"``.

    Returns
    -------
    str
        Full path to the saved checkpoint file.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    return path


def load_checkpoint(path: str, device: torch.device) -> dict:
    """Load a checkpoint from disk.

    Parameters
    ----------
    path : str
        Full path to the ``.pt`` file.
    device : torch.device
        Device onto which tensors are mapped.

    Returns
    -------
    dict
        Checkpoint dictionary as saved by :func:`save_checkpoint`.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


# ─── Timing ─────────────────────────────────────────────────────────────────

class Timer:
    """Context-manager wall-clock timer.

    Example
    -------
    >>> with Timer() as t:
    ...     do_work()
    >>> print(f"Elapsed: {t.elapsed:.2f}s")
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed = time.perf_counter() - self._start


# ─── GPU Warm-up ─────────────────────────────────────────────────────────────

def warmup_gpu(device: torch.device, size: int = 4096) -> None:
    """Perform a small dummy computation to warm up the CUDA runtime.

    This ensures that the first real training step does not include
    JIT / driver initialisation latency in its timing measurement.

    Parameters
    ----------
    device : torch.device
        Target CUDA device.
    size : int
        Side length of the square matrices multiplied during warm-up.
    """
    try:
        a = torch.randn(size, size, device=device, dtype=torch.float32)
        b = torch.randn(size, size, device=device, dtype=torch.float32)
        _ = a @ b
        torch.cuda.synchronize(device)
        del a, b
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"GPU warmup skipped due to device incompatibility: {e}")


if __name__ == "__main__":
    set_seed(42)
    print("Seed set.")

    info = get_gpu_memory_info()
    print(f"GPU free : {format_bytes(info['free'])}")
    print(f"GPU total: {format_bytes(info['total'])}")

    fit = should_cache_on_gpu(n_rows=500_000, n_features=264)
    print(f"Full dataset fits on GPU: {fit}")

    csv_log = CSVLogger("/tmp/tfdmga_test.csv")
    csv_log.log({"epoch": 1, "loss": 0.01, "ic": 0.05})
    csv_log.log({"epoch": 2, "loss": 0.009, "ic": 0.06})
    print("CSVLogger OK.")

    with Timer() as t:
        time.sleep(0.1)
    print(f"Timer OK: {t.elapsed:.3f}s")
