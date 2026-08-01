"""
dataset.py — TFDMGA Financial Dataset
=======================================
Loads ``master_panel_features.parquet`` **exactly once**, cleans it once,
and exposes PyTorch Datasets for each walk-forward split.

GPU caching strategy
---------------------
*   If the full dataset fits in VRAM (with a 60 % safety margin), ALL
    feature tensors are uploaded to the GPU once and mini-batches are
    served from VRAM — eliminating CPU→GPU transfer latency entirely.
*   Otherwise, tensors are kept in pinned (page-locked) host memory and
    streamed asynchronously via ``non_blocking=True`` transfers in the
    Trainer's data-loading loop.

Feature column auto-detection
-------------------------------
The module uses a two-pass heuristic to assign columns to the three
modalities (tech=46, fund=192, macro=26):

  Pass 1 — keyword matching on column names.
  Pass 2 — if counts do not match the spec, it falls back to positional
             slicing after sorting alphabetically.

If ``config.tech_cols / fund_cols / macro_cols`` are explicitly set,
those lists are used directly (no detection).

Author: TFDMGA Research Framework
"""
from __future__ import annotations

import gc
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .config import TFDMGAConfig, WALK_FORWARD_FOLDS, TEST_YEARS
from .utils import (
    format_bytes,
    get_gpu_memory_info,
    setup_logger,
    should_cache_on_gpu,
)


# ─── Feature Detection ────────────────────────────────────────────────────────

_TECH_KEYWORDS = [
    "mom", "momentum", "vol", "volatility", "rsi", "macd", "roc",
    "beta", "alpha", "atr", "bb_", "ema", "sma", "ret_", "return_",
    "price_", "close_", "range_", "stoch", "obv", "adx", "cci",
    "willr", "mfi", "ultosc", "trix", "dmi", "aroon", "kama",
]
_FUND_KEYWORDS = [
    "pe_", "pb_", "ps_", "ev_", "eps", "bvps", "ebitda", "fcf",
    "roe", "roa", "roc", "roic", "gross_margin", "op_margin",
    "net_margin", "debt", "leverage", "asset", "revenue", "income",
    "cashflow", "capex", "dividend", "yield", "payout", "accrual",
    "q_", "quarter", "annual", "fiscal", "book_", "market_cap",
    "sales_", "profit_", "earn_", "fundamental",
]
_MACRO_KEYWORDS = [
    "vix", "yield", "rate", "fed", "treasury", "spread", "credit",
    "oil", "gold", "copper", "dollar", "dxy", "eur", "usd", "gbp",
    "cpi", "ppi", "gdp", "unemployment", "ism", "pmi", "macro",
    "term_", "risk_free", "rf",
]

_IGNORE_COLS = {
    "date", "ticker", "symbol", "permno", "gvkey",
    "target_ret_1d", "target_ret_21d",
    "ret", "ret_1d", "ret_21d", "close", "open", "high", "low",
    "volume", "adj_close", "adj_close_1d",
    "year", "month", "day",
}


_SENT_KEYWORDS = [
    "sentiment", "sent_", "news_", "social_", "mood", "opinion",
]

def _detect_columns(
    df: pd.DataFrame,
    tech_dim: int = 46,
    fund_dim: int = 192,
    macro_dim: int = 26,
    sent_dim: int = 2,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Heuristically assign feature columns to tech / fund / macro / sent groups.

    Parameters
    ----------
    df : pd.DataFrame
        The full loaded dataframe.
    tech_dim, fund_dim, macro_dim, sent_dim : int
        Expected column counts per modality.
    logger : Optional[logging.Logger]

    Returns
    -------
    Tuple[List[str], List[str], List[str], List[str]]
        tech_cols, fund_cols, macro_cols, sent_cols
    """
    # Candidate feature columns (exclude target/meta columns)
    all_feat = [
        c for c in df.columns
        if c.lower() not in {x.lower() for x in _IGNORE_COLS}
        and not any(k in c.lower() for k in ("target", "forward", "fwd", "future"))
    ]

    def _match(col: str, keywords: List[str]) -> bool:
        cl = col.lower()
        return any(k in cl for k in keywords)

    sent_cols  = [c for c in all_feat if _match(c, _SENT_KEYWORDS)]
    macro_cols = [c for c in all_feat if c not in sent_cols and _match(c, _MACRO_KEYWORDS)]
    tech_cols  = [c for c in all_feat if c not in sent_cols and c not in macro_cols and _match(c, _TECH_KEYWORDS)]
    fund_cols  = [c for c in all_feat if c not in sent_cols and c not in macro_cols and c not in tech_cols]

    # ── Fallback: positional split if keyword matching produces wrong counts ──
    if len(tech_cols) != tech_dim or len(fund_cols) != fund_dim or len(macro_cols) != macro_dim or len(sent_cols) != sent_dim:
        if logger:
            logger.warning(
                "Keyword detection produced unexpected column counts: "
                f"tech={len(tech_cols)}/{tech_dim}, "
                f"fund={len(fund_cols)}/{fund_dim}, "
                f"macro={len(macro_cols)}/{macro_dim}, "
                f"sent={len(sent_cols)}/{sent_dim}. "
                "Falling back to positional split."
            )
        sorted_feats = sorted(all_feat)
        total = tech_dim + fund_dim + macro_dim + sent_dim
        if len(sorted_feats) < total:
            raise ValueError(
                f"Dataset has only {len(sorted_feats)} feature columns, "
                f"but {total} are required (tech={tech_dim}, fund={fund_dim}, macro={macro_dim}, sent={sent_dim})."
            )
        tech_cols  = sorted_feats[:tech_dim]
        fund_cols  = sorted_feats[tech_dim : tech_dim + fund_dim]
        macro_cols = sorted_feats[tech_dim + fund_dim : tech_dim + fund_dim + macro_dim]
        sent_cols  = sorted_feats[tech_dim + fund_dim + macro_dim : tech_dim + fund_dim + macro_dim + sent_dim]

    return tech_cols, fund_cols, macro_cols, sent_cols


# ─── Master Data Store ────────────────────────────────────────────────────────

class MasterDataStore:
    """Singleton-like object that loads the parquet file once.

    After construction, ``self.df`` holds the cleaned Pandas DataFrame.
    The heavy NumPy arrays and (optionally) GPU tensors are also cached here.

    Parameters
    ----------
    config : TFDMGAConfig
    logger : Optional[logging.Logger]
    """

    def __init__(
        self,
        config: TFDMGAConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        self.logger = logger or setup_logger("DataStore", config.log_dir)
        self._on_gpu: bool = False
        self._device: Optional[torch.device] = None

        self._load_and_clean(config.data_path)
        self._detect_or_use_explicit_columns()
        self._extract_arrays()

        # Free DataFrame memory immediately to prevent CPU Out of Memory (OOM) crashes
        n_rows = len(self.df)
        del self.df
        import gc
        gc.collect()

        self._maybe_cache_on_gpu(n_rows)

    # ────────────────────────────────────────────────────────────────────────
    def _load_and_clean(self, path: str) -> None:
        """Load parquet once and perform all cleaning in-place."""
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Parquet file not found: {path}\n"
                "Set --data_path or config.data_path to the correct location."
            )
        self.logger.info(f"Loading dataset from: {path}")
        df = pd.read_parquet(path)
        # Downcast float64 to float32 in-place immediately to conserve half the memory
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype(np.float32)
        import gc
        gc.collect()
        self.logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} cols. "
                         f"Memory (optimized): {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

        # ── Mandatory columns check ──────────────────────────────────────────
        for col in ("date", "ticker"):
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in parquet.")

        # ── Date parsing ─────────────────────────────────────────────────────
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year

        # ── Sort ─────────────────────────────────────────────────────────────
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        # ── Target columns ───────────────────────────────────────────────────
        if "target_ret_1d" not in df.columns:
            if "ret" in df.columns:
                df["target_ret_1d"] = df.groupby("ticker")["ret"].shift(-1)
                self.logger.warning("target_ret_1d computed as 1-day forward ret.")
            else:
                raise ValueError(
                    "Column 'target_ret_1d' not found and cannot be derived "
                    "without a 'ret' column."
                )

        if "target_ret_21d" not in df.columns:
            if "ret" in df.columns:
                df["target_ret_21d"] = (
                    df.groupby("ticker")["ret"]
                    .transform(lambda x: x.rolling(21).sum().shift(-21))
                )
                self.logger.warning("target_ret_21d computed as 21-day forward cumsum.")
            else:
                raise ValueError(
                    "Column 'target_ret_21d' not found and cannot be derived "
                    "without a 'ret' column."
                )

        # ── 6-month target (126 trading days) ─────────────────────────────────
        # Aligned with quarterly fundamental data: slow, value-driven signal.
        if "target_ret_126d" not in df.columns:
            if "ret" in df.columns:
                df["target_ret_126d"] = (
                    df.groupby("ticker")["ret"]
                    .transform(lambda x: x.rolling(126).sum().shift(-126))
                )
                self.logger.warning("target_ret_126d computed as 126-day forward cumsum.")
            else:
                raise ValueError(
                    "Column 'target_ret_126d' not found and cannot be derived "
                    "without a 'ret' column."
                )

        # ── Winsorise all three targets (0.1th / 99.9th percentile) ────────────
        for col in ("target_ret_1d", "target_ret_21d", "target_ret_126d"):
            p1  = df[col].quantile(0.001)
            p99 = df[col].quantile(0.999)
            df[col] = df[col].clip(p1, p99)

        # ── Drop rows with NaN in any target ────────────────────────────────
        before = len(df)
        df = df.dropna(
            subset=["target_ret_1d", "target_ret_21d", "target_ret_126d"]
        ).reset_index(drop=True)
        self.logger.info(
            f"Dropped {before - len(df):,} rows with NaN targets. "
            f"Remaining: {len(df):,}"
        )

        self.df = df

    # ────────────────────────────────────────────────────────────────────────
    def _detect_or_use_explicit_columns(self) -> None:
        cfg = self.config
        if cfg.tech_cols and cfg.fund_cols and cfg.macro_cols and cfg.sent_cols:
            self.tech_cols  = cfg.tech_cols
            self.fund_cols  = cfg.fund_cols
            self.macro_cols = cfg.macro_cols
            self.sent_cols  = cfg.sent_cols
            self.logger.info("Using explicit column lists from config.")
        else:
            self.tech_cols, self.fund_cols, self.macro_cols, self.sent_cols = _detect_columns(
                self.df,
                tech_dim=cfg.tech_dim,
                fund_dim=cfg.fund_dim,
                macro_dim=cfg.macro_dim,
                sent_dim=cfg.sent_dim,
                logger=self.logger,
            )

        # Update config dims to match detected counts (handles slight mismatches)
        cfg.tech_dim  = len(self.tech_cols)
        cfg.fund_dim  = len(self.fund_cols)
        cfg.macro_dim = len(self.macro_cols)
        cfg.sent_dim  = len(self.sent_cols)

        self.logger.info(
            f"Feature groups: tech={cfg.tech_dim}, "
            f"fund={cfg.fund_dim}, macro={cfg.macro_dim}, sent={cfg.sent_dim}"
        )

    # ────────────────────────────────────────────────────────────────────────
    def _extract_arrays(self) -> None:
        """Pre-extract feature/target arrays as float32 NumPy arrays."""
        df = self.df
        # Fill NaN with 0 (features assumed cross-sectionally standardised
        # during feature engineering; any remaining NaN → 0)
        self.arr_tech   = df[self.tech_cols].fillna(0).values.astype(np.float32)
        self.arr_fund   = df[self.fund_cols].fillna(0).values.astype(np.float32)
        self.arr_macro  = df[self.macro_cols].fillna(0).values.astype(np.float32)
        self.arr_sent   = df[self.sent_cols].fillna(0).values.astype(np.float32)
        self.arr_y1d    = df["target_ret_1d"].fillna(0).values.astype(np.float32)
        self.arr_y21d   = df["target_ret_21d"].fillna(0).values.astype(np.float32)
        self.arr_y126d  = df["target_ret_126d"].fillna(0).values.astype(np.float32)
        self.arr_year   = df["year"].values.astype(np.int32)
        self.arr_ticker = df["ticker"].values
        self.arr_date   = df["date"].values

        # Pre-compute valid sequence indices (discarding first T-1 days of each stock to prevent ticker leakage)
        T = self.config.window_size
        ticker_arr = df["ticker"].values
        n_rows = len(df)
        self.valid_mask = np.zeros(n_rows, dtype=bool)
        if n_rows >= T:
            self.valid_mask[T - 1:] = (ticker_arr[T - 1:] == ticker_arr[:-T + 1])

        total_feat = len(self.tech_cols) + len(self.fund_cols) + len(self.macro_cols) + len(self.sent_cols)
        n_rows = len(df)
        size_bytes = (n_rows * (total_feat + 3) * 4)  # float32 = 4 bytes, 3 targets
        self.logger.info(
            f"NumPy arrays extracted. "
            f"Rows: {n_rows:,}, Total features: {total_feat}, "
            f"Targets: 1d / 21d / 126d, "
            f"Estimated size: {format_bytes(size_bytes)}"
        )

    # ────────────────────────────────────────────────────────────────────────
    def _maybe_cache_on_gpu(self, n_rows: int) -> None:
        """Upload all arrays to GPU VRAM if they fit comfortably."""
        if not torch.cuda.is_available():
            self.logger.info("No CUDA device found. Using pinned-memory streaming.")
            self._pin_arrays()
            return

        total_feat = self.config.tech_dim + self.config.fund_dim + self.config.macro_dim + self.config.sent_dim
        device = torch.device("cuda:0")

        if should_cache_on_gpu(n_rows, total_feat, n_targets=3, device=device):
            self.logger.info("Dataset fits in VRAM → uploading to GPU for zero-copy serving.")
            self._device = device
            self._on_gpu = True
            self.t_tech   = torch.from_numpy(self.arr_tech).to(device, non_blocking=True)
            self.t_fund   = torch.from_numpy(self.arr_fund).to(device, non_blocking=True)
            self.t_macro  = torch.from_numpy(self.arr_macro).to(device, non_blocking=True)
            self.t_sent   = torch.from_numpy(self.arr_sent).to(device, non_blocking=True)
            self.t_y1d    = torch.from_numpy(self.arr_y1d).to(device, non_blocking=True)
            self.t_y21d   = torch.from_numpy(self.arr_y21d).to(device, non_blocking=True)
            self.t_y126d  = torch.from_numpy(self.arr_y126d).to(device, non_blocking=True)
            torch.cuda.synchronize(device)
            info = get_gpu_memory_info(device)
            self.logger.info(
                f"GPU cache loaded. "
                f"VRAM used: {format_bytes(info['used'])} / {format_bytes(info['total'])}"
            )
        else:
            self.logger.info(
                "Dataset too large for VRAM → using pinned-memory + async transfers."
            )
            self._pin_arrays()

    def _pin_arrays(self) -> None:
        """Convert NumPy arrays to pinned-memory CPU tensors for fast DMA transfer."""
        try:
            self.t_tech   = torch.from_numpy(self.arr_tech).pin_memory()
            self.t_fund   = torch.from_numpy(self.arr_fund).pin_memory()
            self.t_macro  = torch.from_numpy(self.arr_macro).pin_memory()
            self.t_sent   = torch.from_numpy(self.arr_sent).pin_memory()
            self.t_y1d    = torch.from_numpy(self.arr_y1d).pin_memory()
            self.t_y21d   = torch.from_numpy(self.arr_y21d).pin_memory()
            self.t_y126d  = torch.from_numpy(self.arr_y126d).pin_memory()
        except Exception as e:
            self.logger.warning(f"pin_memory() failed: {e}. Falling back to standard CPU tensors.")
            self.t_tech   = torch.from_numpy(self.arr_tech)
            self.t_fund   = torch.from_numpy(self.arr_fund)
            self.t_macro  = torch.from_numpy(self.arr_macro)
            self.t_sent   = torch.from_numpy(self.arr_sent)
            self.t_y1d    = torch.from_numpy(self.arr_y1d)
            self.t_y21d   = torch.from_numpy(self.arr_y21d)
            self.t_y126d  = torch.from_numpy(self.arr_y126d)


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class FinancialPanelDataset(Dataset):
    """PyTorch Dataset wrapping a year-filtered view of the MasterDataStore.

    Parameters
    ----------
    store : MasterDataStore
        Pre-loaded data store (shared across folds — no re-loading).
    years : List[int]
        Year subset to expose (e.g. ``[2015, 2016, 2017, 2018]``).
    """

    def __init__(
        self,
        store: MasterDataStore,
        years: List[int],
    ) -> None:
        super().__init__()
        self.store = store

        # Build index mask for the requested years and ensure valid sequence window
        mask = np.isin(store.arr_year, years) & store.valid_mask
        self.indices = np.where(mask)[0].astype(np.int64)

        n = len(self.indices)
        if n == 0:
            raise ValueError(
                f"No data found for years {years}. "
                f"Dataset covers years: {sorted(set(store.arr_year))}"
            )

        self.logger = store.logger
        self.logger.debug(f"Dataset for years={years}: {n:,} samples.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, ...]:
        """Return (x_tech, x_fund, x_macro, y_1d, y_21d, y_126d) for sample ``i``.

        When the store is GPU-cached, tensors are sliced directly from VRAM
        (zero-copy indexing).  When pinned, tensors are sliced from pinned
        host memory; the Trainer transfers them to GPU with ``non_blocking=True``.

        The three targets correspond to:
          y_1d   — 1-day forward return   (technical signal horizon)
          y_21d  — 21-day forward return  (macro signal horizon)
          y_126d — 126-day forward return (fundamental signal horizon)
        """
        idx = int(self.indices[i])
        T = self.store.config.window_size
        store = self.store

        # O(T) backward ticker verification search
        target_ticker = store.arr_ticker[idx]
        for offset in range(T):
            if store.arr_ticker[idx - offset] != target_ticker:
                raise ValueError(
                    f"Ticker leakage detected! Index {idx - offset} has ticker "
                    f"'{store.arr_ticker[idx - offset]}' which does not match target ticker "
                    f"'{target_ticker}' at index {idx}."
                )

        x_tech  = store.t_tech[idx - T + 1 : idx + 1]
        x_fund  = store.t_fund[idx - T + 1 : idx + 1]
        x_macro = store.t_macro[idx - T + 1 : idx + 1]
        x_sent  = store.t_sent[idx - T + 1 : idx + 1]
        y_1d    = store.t_y1d[idx].unsqueeze(0)
        y_21d   = store.t_y21d[idx].unsqueeze(0)
        y_126d  = store.t_y126d[idx].unsqueeze(0)
        return x_tech, x_fund, x_macro, x_sent, y_1d, y_21d, y_126d

    @property
    def df(self) -> pd.DataFrame:
        """Return a lightweight DataFrame slice for this subset (for metric computation)."""
        data = {
            "date": self.store.arr_date[self.indices],
            "year": self.store.arr_year[self.indices],
            "ticker": self.store.arr_ticker[self.indices],
            "target_ret_1d": self.store.arr_y1d[self.indices],
            "target_ret_21d": self.store.arr_y21d[self.indices],
            "target_ret_126d": self.store.arr_y126d[self.indices],
        }
        return pd.DataFrame(data)


# ─── DataLoader Factory ───────────────────────────────────────────────────────

def make_dataloader(
    dataset: FinancialPanelDataset,
    batch_size: int,
    shuffle: bool,
    config: TFDMGAConfig,
) -> DataLoader:
    """Construct an optimised DataLoader for the given dataset.

    When the dataset is GPU-cached (``store._on_gpu == True``), we use
    ``num_workers=0`` and no pinning (data is already on GPU).
    Otherwise, we use persistent workers, pinned memory, and prefetching.

    Parameters
    ----------
    dataset : FinancialPanelDataset
    batch_size : int
    shuffle : bool
    config : TFDMGAConfig

    Returns
    -------
    DataLoader
    """
    if dataset.store._on_gpu:
        # Data already on GPU — single-process is fastest (no IPC overhead)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
            drop_last=shuffle,
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers and config.num_workers > 0,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
            drop_last=shuffle,
        )


# ─── Walk-Forward Splitter ────────────────────────────────────────────────────

class WalkForwardSplitter:
    """Generates expanding-window train/val dataset pairs for each fold.

    Parameters
    ----------
    store : MasterDataStore
    config : TFDMGAConfig
    """

    def __init__(self, store: MasterDataStore, config: TFDMGAConfig) -> None:
        self.store  = store
        self.config = config

    def get_fold(
        self, fold_idx: int
    ) -> Tuple[FinancialPanelDataset, FinancialPanelDataset]:
        """Return (train_dataset, val_dataset) for a specific fold (1-indexed).

        Parameters
        ----------
        fold_idx : int
            Fold index in [1, len(WALK_FORWARD_FOLDS)].

        Returns
        -------
        Tuple[FinancialPanelDataset, FinancialPanelDataset]
        """
        fold_def = WALK_FORWARD_FOLDS[fold_idx - 1]
        train_ds = FinancialPanelDataset(self.store, years=fold_def["train_years"])
        val_ds   = FinancialPanelDataset(self.store, years=fold_def["val_years"])
        return train_ds, val_ds

    def get_test(self) -> FinancialPanelDataset:
        """Return the final hold-out test dataset (2024 only)."""
        return FinancialPanelDataset(self.store, years=TEST_YEARS)

    def n_folds(self) -> int:
        return len(WALK_FORWARD_FOLDS)


# ─── Self-contained smoke test ────────────────────────────────────────────────

if __name__ == "__main__":
    """Smoke-test with a synthetic in-memory parquet written to /tmp."""
    import tempfile, os

    rng = np.random.default_rng(42)
    n = 50_000
    dates   = pd.date_range("2015-01-01", periods=n // 20, freq="B").repeat(20)
    tickers = (["TICK%03d" % i for i in range(20)] * (n // 20))[:n]

    cols: Dict[str, np.ndarray] = {"date": dates[:n], "ticker": tickers}
    for i in range(46):
        cols[f"tech_{i:02d}"] = rng.normal(0, 1, n).astype(np.float32)
    for i in range(192):
        cols[f"fund_{i:03d}"] = rng.normal(0, 1, n).astype(np.float32)
    for i in range(26):
        cols[f"macro_{i:02d}"] = rng.normal(0, 1, n).astype(np.float32)
    for i in range(2):
        cols[f"sentiment_{i:02d}"] = rng.normal(0, 1, n).astype(np.float32)
    cols["target_ret_1d"]  = rng.normal(0, 0.01, n).astype(np.float32)
    cols["target_ret_21d"] = rng.normal(0, 0.05, n).astype(np.float32)
    cols["target_ret_126d"] = rng.normal(0, 0.1, n).astype(np.float32)

    df_synth = pd.DataFrame(cols)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = f.name
    df_synth.to_parquet(tmp_path, index=False)
    print(f"Synthetic parquet written: {tmp_path}")

    cfg = TFDMGAConfig(
        data_path=tmp_path,
        tech_dim=46, fund_dim=192, macro_dim=26, sent_dim=2,
        checkpoint_dir="/tmp/tfdmga_ckpt",
        log_dir="/tmp/tfdmga_logs",
        results_dir="/tmp/tfdmga_results",
        num_workers=0,
    )

    store = MasterDataStore(cfg)
    print(f"Store loaded: {len(store.arr_year):,} rows")

    splitter = WalkForwardSplitter(store, cfg)
    train_ds, val_ds = splitter.get_fold(1)
    print(f"Fold 1: train={len(train_ds):,}, val={len(val_ds):,}")

    loader = make_dataloader(train_ds, batch_size=256, shuffle=True, config=cfg)
    batch  = next(iter(loader))
    xt, xf, xm, xs, y1, y21, y126 = batch
    print(f"Batch shapes: tech={tuple(xt.shape)}, fund={tuple(xf.shape)}, "
          f"macro={tuple(xm.shape)}, sent={tuple(xs.shape)}, y1d={tuple(y1.shape)}, y126d={tuple(y126.shape)}")

    os.unlink(tmp_path)
    print("Dataset smoke test PASSED.")
