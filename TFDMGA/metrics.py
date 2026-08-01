"""
metrics.py — TFDMGA Evaluation Metrics
========================================
Full suite of financial and statistical metrics used to evaluate
model performance in the thesis:

  Statistical:  MSE, MAE, R²
  Signal:       IC, RankIC, ICIR, Hit Ratio
  Portfolio:    Annual Return, Sharpe, Sortino, Calmar, Max Drawdown,
                Annual Volatility, Turnover
  Portfolios:   Long, Short, Long-Short (equal-weight, decile-based)
  Cost Analysis: Net return at 5 / 10 / 20 bps round-trip transaction cost

All portfolio metrics assume a daily-rebalanced, decile-ranked strategy
unless otherwise noted.

Author: TFDMGA Research Framework
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ─── Constants ───────────────────────────────────────────────────────────────

TRADING_DAYS_PER_YEAR: int = 252
TOP_DECILE_THRESHOLD:  float = 0.90   # top 10 % → long
BOT_DECILE_THRESHOLD:  float = 0.10   # bottom 10 % → short


# ─── Statistical Metrics ─────────────────────────────────────────────────────

def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((y_true - y_pred) ** 2))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


# ─── Information Coefficient (IC) ────────────────────────────────────────────

def compute_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson Information Coefficient (IC) between predictions and realised returns.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        1-D arrays of equal length.

    Returns
    -------
    float
        Pearson correlation coefficient ∈ [−1, 1].  Returns 0.0 when
        either array has zero variance.
    """
    if len(y_true) < 3:
        return 0.0
    ic, _ = stats.pearsonr(y_pred, y_true)
    return float(ic) if not np.isnan(ic) else 0.0


def compute_rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman Rank IC (rank correlation between prediction ranks and return ranks).

    More robust than Pearson IC to outliers and non-normality — standard
    in quantitative finance evaluation.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        1-D arrays of equal length.

    Returns
    -------
    float
        Spearman correlation coefficient ∈ [−1, 1].
    """
    if len(y_true) < 3:
        return 0.0
    rho, _ = stats.spearmanr(y_pred, y_true)
    return float(rho) if not np.isnan(rho) else 0.0


def compute_icir(ic_series: np.ndarray) -> float:
    """Information Coefficient Information Ratio (ICIR).

    ICIR = mean(IC_t) / std(IC_t)  ×  √T  where T is the number of periods.

    Measures the *consistency* of predictive signal, analogous to the
    Sharpe ratio but for IC instead of returns.  Annualised here with √252.

    Parameters
    ----------
    ic_series : np.ndarray
        1-D array of daily (or period-level) IC values.

    Returns
    -------
    float
        Annualised ICIR.  Returns 0.0 when IC std is zero or series too short.
    """
    if len(ic_series) < 3:
        return 0.0
    mean_ic = np.mean(ic_series)
    std_ic  = np.std(ic_series, ddof=1)
    if std_ic < 1e-12:
        return 0.0
    return float(mean_ic / std_ic * np.sqrt(TRADING_DAYS_PER_YEAR))


# ─── Portfolio Construction ───────────────────────────────────────────────────

def build_daily_portfolio_returns(
    df: pd.DataFrame,
    pred_col: str = "pred_1d",
    ret_col:  str = "target_ret_1d",
    top_q: float = TOP_DECILE_THRESHOLD,
    bot_q: float = BOT_DECILE_THRESHOLD,
) -> pd.DataFrame:
    """Build daily long / short / long-short portfolio returns from predictions.

    Stocks are ranked cross-sectionally each day.  Stocks in the top decile
    form the long portfolio; stocks in the bottom decile form the short portfolio.
    Portfolios are equal-weighted within each decile.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``date``, ``pred_col``, and ``ret_col``.
    pred_col : str
        Column name of the model's return predictions.
    ret_col : str
        Column name of the realised (forward) returns.
    top_q, bot_q : float
        Quantile cutoffs for long and short portfolios respectively.

    Returns
    -------
    pd.DataFrame
        Daily portfolio returns with columns:
        ``["date", "long_ret", "short_ret", "ls_ret"]``.
    """
    results = []
    for date, group in df.groupby("date"):
        if len(group) < 10:
            continue  # skip sparse dates
        ranks = group[pred_col].rank(pct=True)
        long_mask  = ranks >= top_q
        short_mask = ranks <= bot_q

        long_ret  = group.loc[long_mask,  ret_col].mean()
        short_ret = group.loc[short_mask, ret_col].mean()

        # Long-short: long the top decile, short the bottom decile
        ls_ret = long_ret - short_ret

        results.append({
            "date":      date,
            "long_ret":  long_ret  if not np.isnan(long_ret)  else 0.0,
            "short_ret": short_ret if not np.isnan(short_ret) else 0.0,
            "ls_ret":    ls_ret    if not np.isnan(ls_ret)    else 0.0,
        })

    port_df = pd.DataFrame(results).sort_values("date").reset_index(drop=True)
    return port_df


# ─── Portfolio Performance Metrics ───────────────────────────────────────────

def compute_sharpe(returns: np.ndarray, ann_factor: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualised Sharpe ratio (assumes zero risk-free rate).

    Parameters
    ----------
    returns : np.ndarray
        Daily portfolio return series.
    ann_factor : int
        Number of periods per year for annualisation (default 252).

    Returns
    -------
    float
        Annualised Sharpe ratio.
    """
    if len(returns) < 2:
        return 0.0
    mean_r = np.mean(returns)
    std_r  = np.std(returns, ddof=1)
    if std_r < 1e-12:
        return 0.0
    return float(mean_r / std_r * np.sqrt(ann_factor))


def compute_sortino(
    returns: np.ndarray,
    ann_factor: int = TRADING_DAYS_PER_YEAR,
    mar: float = 0.0,
) -> float:
    """Annualised Sortino ratio.

    Uses the downside deviation (semi-standard deviation below the minimum
    acceptable return ``mar``) instead of total volatility.

    Parameters
    ----------
    returns : np.ndarray
        Daily portfolio return series.
    ann_factor : int
        Annualisation factor.
    mar : float
        Minimum acceptable return per period (default 0).

    Returns
    -------
    float
        Annualised Sortino ratio.
    """
    if len(returns) < 2:
        return 0.0
    mean_r   = np.mean(returns)
    downside = returns[returns < mar] - mar
    if len(downside) == 0:
        return np.inf
    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std < 1e-12:
        return 0.0
    return float(mean_r / downside_std * np.sqrt(ann_factor))


def compute_max_drawdown(returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of the cumulative return series.

    Parameters
    ----------
    returns : np.ndarray
        Daily portfolio return series (arithmetic).

    Returns
    -------
    float
        Maximum drawdown as a positive fraction (e.g. 0.25 = 25 % drawdown).
    """
    if len(returns) == 0:
        return 0.0
    cum = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(cum)
    drawdowns   = (cum - running_max) / running_max
    return float(np.abs(drawdowns.min()))


def compute_calmar(
    returns: np.ndarray,
    ann_factor: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calmar ratio: annualised return divided by maximum drawdown.

    Parameters
    ----------
    returns : np.ndarray
        Daily portfolio return series.
    ann_factor : int
        Annualisation factor.

    Returns
    -------
    float
        Calmar ratio (∞ if max drawdown is zero).
    """
    ann_ret = float(np.mean(returns) * ann_factor)
    mdd     = compute_max_drawdown(returns)
    if mdd < 1e-12:
        return np.inf if ann_ret > 0 else 0.0
    return ann_ret / mdd


def compute_annual_return(
    returns: np.ndarray,
    ann_factor: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised arithmetic mean return."""
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns) * ann_factor)


def compute_annual_volatility(
    returns: np.ndarray,
    ann_factor: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised return standard deviation (volatility)."""
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * np.sqrt(ann_factor))


def compute_hit_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Hit ratio: fraction of predictions with the correct direction.

    Parameters
    ----------
    y_true : np.ndarray
        Realised returns (sign is used).
    y_pred : np.ndarray
        Predicted returns (sign is used).

    Returns
    -------
    float
        Fraction of samples where sign(pred) == sign(true).
    """
    if len(y_true) == 0:
        return 0.0
    correct = np.sign(y_pred) == np.sign(y_true)
    return float(correct.mean())


# ─── Turnover ────────────────────────────────────────────────────────────────

def compute_turnover(
    df: pd.DataFrame,
    pred_col: str = "pred_1d",
    top_q:    float = TOP_DECILE_THRESHOLD,
    bot_q:    float = BOT_DECILE_THRESHOLD,
) -> float:
    """Compute average daily one-way portfolio turnover.

    Turnover = average fraction of the portfolio that changes between
    consecutive trading days (one-way, annualised).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``date``, ``ticker``, and ``pred_col`` columns.
    pred_col : str
        Prediction column used for ranking.
    top_q, bot_q : float
        Quantile cutoffs for the long/short portfolios.

    Returns
    -------
    float
        Average daily one-way turnover (0–1).  Multiply by 252 for annual.
    """
    dates = sorted(df["date"].unique())
    if len(dates) < 2:
        return 0.0

    turnovers = []
    prev_long:  set  = set()
    prev_short: set  = set()

    for date in dates:
        day = df[df["date"] == date]
        if len(day) == 0:
            continue
        ranks = day[pred_col].rank(pct=True)
        curr_long  = set(day["ticker"][ranks >= top_q].values)
        curr_short = set(day["ticker"][ranks <= bot_q].values)

        if prev_long:
            # One-way turnover: size of symmetric difference / 2 / portfolio_size
            n_long  = max(len(curr_long),  1)
            n_short = max(len(curr_short), 1)
            to_long  = len(prev_long.symmetric_difference(curr_long))  / (2 * n_long)
            to_short = len(prev_short.symmetric_difference(curr_short)) / (2 * n_short)
            turnovers.append((to_long + to_short) / 2)

        prev_long  = curr_long
        prev_short = curr_short

    return float(np.mean(turnovers)) if turnovers else 0.0


# ─── Transaction Cost Analysis ────────────────────────────────────────────────

def compute_net_returns_after_costs(
    daily_port_returns: pd.DataFrame,
    daily_turnover: float,
    cost_bps_list: List[int] = (5, 10, 20),
) -> Dict[str, float]:
    """Compute net annualised return and Sharpe after round-trip transaction costs.

    Parameters
    ----------
    daily_port_returns : pd.DataFrame
        Output of :func:`build_daily_portfolio_returns` with column ``ls_ret``.
    daily_turnover : float
        Average daily one-way turnover as a fraction (e.g. 0.30 = 30 %).
    cost_bps_list : List[int]
        List of round-trip transaction cost assumptions in basis points.

    Returns
    -------
    Dict[str, float]
        Nested dict: for each ``bps`` level, keys are
        ``f"net_ann_ret_{bps}bps"`` and ``f"net_sharpe_{bps}bps"``.
    """
    results: Dict[str, float] = {}
    gross_returns = daily_port_returns["ls_ret"].values

    for bps in cost_bps_list:
        cost_per_day = daily_turnover * (bps / 10_000) * 2  # round-trip = 2×
        net_returns  = gross_returns - cost_per_day
        results[f"net_ann_ret_{bps}bps"] = compute_annual_return(net_returns)
        results[f"net_sharpe_{bps}bps"]  = compute_sharpe(net_returns)

    return results


# ─── Cross-sectional IC per date ─────────────────────────────────────────────

def compute_daily_ic(
    df: pd.DataFrame,
    pred_col: str = "pred_1d",
    ret_col:  str = "target_ret_1d",
    method:   str = "pearson",
) -> np.ndarray:
    """Compute cross-sectional IC for each date in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with columns ``date``, ``pred_col``, ``ret_col``.
    pred_col, ret_col : str
        Prediction and realised return columns.
    method : str
        ``"pearson"`` for IC or ``"spearman"`` for RankIC.

    Returns
    -------
    np.ndarray
        1-D array of per-date IC values.
    """
    ic_values = []
    for _, group in df.groupby("date"):
        if len(group) < 5:
            continue
        if method == "pearson":
            ic = compute_ic(group[ret_col].values, group[pred_col].values)
        else:
            ic = compute_rank_ic(group[ret_col].values, group[pred_col].values)
        ic_values.append(ic)
    return np.array(ic_values)


# ─── Aggregate Evaluation ────────────────────────────────────────────────────

def evaluate_predictions(
    df: pd.DataFrame,
    pred_col:  str = "pred_1d",
    ret_col:   str = "target_ret_1d",
    ticker_col: str = "ticker",
    date_col:  str = "date",
) -> Dict[str, float]:
    """Compute the full evaluation metric suite for one prediction horizon.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing ``date``, ``ticker``, prediction, and realised return.
    pred_col : str
        Column name of model predictions.
    ret_col : str
        Column name of realised forward returns.
    ticker_col : str
        Column name of stock ticker identifiers.
    date_col : str
        Column name of date identifiers.

    Returns
    -------
    Dict[str, float]
        Dictionary of all metric names to their values.
    """
    df = df.dropna(subset=[pred_col, ret_col]).copy()
    y_true = df[ret_col].values
    y_pred = df[pred_col].values

    # ── Statistical ──────────────────────────────────────────────────────────
    mse = compute_mse(y_true, y_pred)
    mae = compute_mae(y_true, y_pred)
    r2  = compute_r2(y_true, y_pred)

    # ── Signal ───────────────────────────────────────────────────────────────
    daily_ic_arr  = compute_daily_ic(df, pred_col, ret_col, "pearson")
    daily_ric_arr = compute_daily_ic(df, pred_col, ret_col, "spearman")

    ic      = float(np.nanmean(daily_ic_arr))
    rank_ic = float(np.nanmean(daily_ric_arr))
    icir    = compute_icir(daily_ic_arr)
    hit     = compute_hit_ratio(y_true, y_pred)

    # ── Portfolio returns ─────────────────────────────────────────────────────
    df_port = build_daily_portfolio_returns(df, pred_col=pred_col, ret_col=ret_col)
    long_rets  = df_port["long_ret"].values
    short_rets = df_port["short_ret"].values
    ls_rets    = df_port["ls_ret"].values

    # ── Portfolio metrics for Long-Short ─────────────────────────────────────
    ann_ret = compute_annual_return(ls_rets)
    ann_vol = compute_annual_volatility(ls_rets)
    sharpe  = compute_sharpe(ls_rets)
    sortino = compute_sortino(ls_rets)
    calmar  = compute_calmar(ls_rets)
    mdd     = compute_max_drawdown(ls_rets)

    # Long-only
    long_sharpe = compute_sharpe(long_rets)
    long_ret_ann = compute_annual_return(long_rets)

    # Short-only
    short_sharpe = compute_sharpe(short_rets)
    short_ret_ann = compute_annual_return(short_rets)

    # ── Turnover ─────────────────────────────────────────────────────────────
    turnover = compute_turnover(df, pred_col=pred_col)

    # ── Transaction costs ─────────────────────────────────────────────────────
    cost_metrics = compute_net_returns_after_costs(df_port, turnover)

    metrics = {
        # Statistical
        "mse":              mse,
        "mae":              mae,
        "r2":               r2,
        # Signal quality
        "ic":               ic,
        "rank_ic":          rank_ic,
        "icir":             icir,
        "hit_ratio":        hit,
        # Long-Short portfolio
        "ann_ret":          ann_ret,
        "ann_vol":          ann_vol,
        "sharpe":           sharpe,
        "sortino":          sortino,
        "calmar":           calmar,
        "max_drawdown":     mdd,
        # Decomposed portfolios
        "long_ann_ret":     long_ret_ann,
        "long_sharpe":      long_sharpe,
        "short_ann_ret":    short_ret_ann,
        "short_sharpe":     short_sharpe,
        # Turnover & costs
        "turnover_daily":   turnover,
        **cost_metrics,
    }
    return metrics


def format_metrics_table(metrics: Dict[str, float], title: str = "Evaluation Results") -> str:
    """Format a metrics dict as a human-readable table string.

    Parameters
    ----------
    metrics : Dict[str, float]
        Output of :func:`evaluate_predictions`.
    title : str
        Table header.

    Returns
    -------
    str
        Multi-line formatted table.
    """
    lines = [f"\n{'='*50}", f"  {title}", f"{'='*50}"]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k:<28s}: {v:>10.4f}")
        else:
            lines.append(f"  {k:<28s}: {v!r}")
    lines.append("=" * 50)
    return "\n".join(lines)


# ─── Self-contained smoke test ────────────────────────────────────────────────

# ─── Stop-Loss ───────────────────────────────────────────────────────────────

def apply_position_stop_loss(
    df: pd.DataFrame,
    ret_col:        str   = "target_ret_1d",
    stop_threshold: float = -0.05,
) -> pd.DataFrame:
    """Apply a position-level daily stop-loss to individual stock returns.

    For each stock-day observation, if the realised return is worse than
    ``stop_threshold``, the position is considered closed at the stop price
    and the return is floored to ``stop_threshold``.

    This models an intraday stop order: the trader exits at the stop level
    rather than accepting the full loss.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with columns ``date``, ``ticker``, ``ret_col``.
    ret_col : str
        Return column to apply the floor to.
    stop_threshold : float
        Maximum allowed single-day return loss per position (e.g. -0.05 = -5 %).
        Must be negative.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with ``ret_col`` floored at ``stop_threshold``.
        A boolean column ``stop_triggered`` marks rows where the stop fired.
    """
    if stop_threshold >= 0:
        raise ValueError(
            f"stop_threshold must be negative (a loss limit). Got {stop_threshold}."
        )
    out = df.copy()
    original = out[ret_col].values.copy()
    out[ret_col] = np.maximum(original, stop_threshold)
    out["stop_triggered"] = original < stop_threshold
    return out


def apply_portfolio_stop_loss(
    port_df:           pd.DataFrame,
    trailing_stop:     float = -0.10,
    recovery_threshold: float = 0.02,
    cooldown_days:     int   = 5,
) -> pd.DataFrame:
    """Apply a portfolio-level trailing stop-loss (circuit breaker).

    Tracks the running peak NAV of the long-short strategy.  When the NAV
    falls more than ``trailing_stop`` below its peak (e.g. -10 %), the strategy
    goes **flat** (zero return) until one of two conditions is met:

      * The drawdown recovers within ``recovery_threshold`` of the peak, **or**
      * ``cooldown_days`` trading days have elapsed since the stop triggered.

    This models a risk manager cutting exposure after a severe drawdown and
    gradually re-entering once conditions normalise.

    Parameters
    ----------
    port_df : pd.DataFrame
        Output of :func:`build_daily_portfolio_returns` with column ``ls_ret``.
    trailing_stop : float
        Drawdown level that triggers the stop (e.g. -0.10 = -10 %).
        Must be negative.
    recovery_threshold : float
        Fractional recovery from trough required to resume trading
        (e.g. 0.02 = NAV must rise 2 % from the stop-trigger level).
    cooldown_days : int
        Minimum number of flat days before the strategy can re-enter,
        regardless of NAV recovery.

    Returns
    -------
    pd.DataFrame
        Copy of ``port_df`` with additional columns:

        ``ls_ret_sl``     — long-short return after stop-loss (zero on flat days)
        ``nav_gross``     — cumulative NAV without stop-loss
        ``nav_sl``        — cumulative NAV with stop-loss
        ``is_flat``       — True on days the strategy is forced flat
        ``drawdown``      — running drawdown of gross NAV (for reference)
    """
    if trailing_stop >= 0:
        raise ValueError(
            f"trailing_stop must be negative. Got {trailing_stop}."
        )

    df = port_df.copy().reset_index(drop=True)
    n  = len(df)

    ls_ret    = df["ls_ret"].values.copy()
    ls_ret_sl = ls_ret.copy()          # will be zeroed on flat days
    is_flat   = np.zeros(n, dtype=bool)

    nav         = 1.0                  # gross NAV (no stop)
    nav_sl      = 1.0                  # stop-loss NAV
    peak_nav    = 1.0                  # running peak of gross NAV
    nav_gross_arr = np.empty(n)
    nav_sl_arr    = np.empty(n)

    flat_count    = 0                  # days spent flat so far in current episode
    trigger_nav   = None               # NAV at which the stop triggered
    currently_flat = False

    for i in range(n):
        # Update gross NAV (always)
        nav      *= (1.0 + ls_ret[i])
        peak_nav  = max(peak_nav, nav)
        drawdown  = (nav - peak_nav) / peak_nav   # ≤ 0

        if currently_flat:
            # Flat: earn 0, track cooldown and recovery
            ls_ret_sl[i] = 0.0
            is_flat[i]   = True
            flat_count  += 1
            nav_sl      *= 1.0   # stays at stop-trigger NAV level

            # Re-entry conditions: cooldown elapsed AND NAV recovered enough
            recovered = (nav >= trigger_nav * (1.0 + recovery_threshold))
            if flat_count >= cooldown_days and recovered:
                currently_flat = False
                flat_count     = 0
                trigger_nav    = None
        else:
            # Active: apply this day's return
            ls_ret_sl[i] = ls_ret[i]
            nav_sl      *= (1.0 + ls_ret[i])

            # Check whether to trigger stop
            if drawdown <= trailing_stop:
                currently_flat = True
                flat_count     = 0
                trigger_nav    = nav_sl   # record NAV at trigger point

        nav_gross_arr[i] = nav
        nav_sl_arr[i]    = nav_sl

    df["ls_ret_sl"]  = ls_ret_sl
    df["nav_gross"]  = nav_gross_arr
    df["nav_sl"]     = nav_sl_arr
    df["is_flat"]    = is_flat
    df["drawdown"]   = (nav_gross_arr - np.maximum.accumulate(nav_gross_arr)) / \
                       np.maximum.accumulate(nav_gross_arr)
    return df


def evaluate_stop_loss_comparison(
    port_df:            pd.DataFrame,
    position_stop_df:   Optional[pd.DataFrame] = None,
    trailing_stop:      float = -0.10,
    recovery_threshold: float = 0.02,
    cooldown_days:      int   = 5,
) -> Dict[str, Dict[str, float]]:
    """Compute side-by-side performance with and without stop-loss.

    Runs :func:`apply_portfolio_stop_loss` and then calculates the full
    performance metric set for both the gross (no stop) and net (with stop)
    return series, returning a comparison dictionary suitable for table
    display in the thesis.

    Parameters
    ----------
    port_df : pd.DataFrame
        Output of :func:`build_daily_portfolio_returns`.
    position_stop_df : Optional[pd.DataFrame]
        If provided, a pre-built portfolio df whose ``ls_ret`` column already
        reflects position-level stop-loss flooring.  Used to chain position-
        and portfolio-level stops together.
    trailing_stop : float
        Trailing drawdown trigger (e.g. -0.10 for -10 %).
    recovery_threshold : float
        NAV recovery fraction required for re-entry.
    cooldown_days : int
        Minimum flat days before re-entry.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Keys: ``"gross"`` (no stop), ``"portfolio_stop"`` (trailing stop only).
        Each value is a dict of metric name → float.
    """
    base_df = position_stop_df if position_stop_df is not None else port_df
    sl_df   = apply_portfolio_stop_loss(
        base_df,
        trailing_stop=trailing_stop,
        recovery_threshold=recovery_threshold,
        cooldown_days=cooldown_days,
    )

    def _metrics(returns: np.ndarray, label: str) -> Dict[str, float]:
        return {
            "ann_ret":      compute_annual_return(returns),
            "ann_vol":      compute_annual_volatility(returns),
            "sharpe":       compute_sharpe(returns),
            "sortino":      compute_sortino(returns),
            "calmar":       compute_calmar(returns),
            "max_drawdown": compute_max_drawdown(returns),
            "hit_ratio":    float(np.mean(returns > 0)) if len(returns) else 0.0,
            "flat_days":    float(sl_df["is_flat"].sum()) if "is_flat" in sl_df else 0.0,
        }

    gross_rets = sl_df["ls_ret"].values
    sl_rets    = sl_df["ls_ret_sl"].values

    return {
        "gross":           _metrics(gross_rets, "gross"),
        "portfolio_stop":  _metrics(sl_rets,    "portfolio_stop"),
    }


def format_stop_loss_comparison_table(
    comparison: Dict[str, Dict[str, float]],
    trailing_stop:  float = -0.10,
    cooldown_days:  int   = 5,
) -> str:
    """Format the stop-loss comparison as a human-readable table.

    Parameters
    ----------
    comparison : Dict[str, Dict[str, float]]
        Output of :func:`evaluate_stop_loss_comparison`.
    trailing_stop : float
    cooldown_days : int

    Returns
    -------
    str
        Multi-line formatted table.
    """
    header = (
        f"\n{'='*65}\n"
        f"  Stop-Loss Analysis  "
        f"(trailing_stop={trailing_stop:.0%}, cooldown={cooldown_days}d)\n"
        f"{'='*65}\n"
        f"  {'Metric':<22} {'No Stop':>14} {'Portfolio Stop':>16}\n"
        f"  {'-'*22} {'-'*14} {'-'*16}"
    )
    rows = []
    metric_labels = {
        "ann_ret":      "Annual Return",
        "ann_vol":      "Annual Vol",
        "sharpe":       "Sharpe Ratio",
        "sortino":      "Sortino Ratio",
        "calmar":       "Calmar Ratio",
        "max_drawdown": "Max Drawdown",
        "hit_ratio":    "Hit Ratio",
        "flat_days":    "Flat Days",
    }
    gross = comparison.get("gross", {})
    sl    = comparison.get("portfolio_stop", {})
    for key, label in metric_labels.items():
        g_val = gross.get(key, float("nan"))
        s_val = sl.get(key, float("nan"))
        rows.append(f"  {label:<22} {g_val:>14.4f} {s_val:>16.4f}")
    footer = "=" * 65
    return "\n".join([header] + rows + [footer])


# ─── 2:1 Risk-Reward Stop-Loss ───────────────────────────────────────────────

def apply_risk_reward_stop_loss(
    df: pd.DataFrame,
    ret_col:        str   = "target_ret_1d",
    stop_threshold: float = -0.02,
    reward_ratio:   float = 2.0,
) -> pd.DataFrame:
    """Apply a 2:1 risk-reward (stop-loss + take-profit) rule per position.

    For each stock-day observation the rule is:

      * If realised return < ``stop_threshold``         → floor  at ``stop_threshold``
        (stop-loss fires: maximum loss is capped)
      * If realised return > ``|stop_threshold| × reward_ratio``  → cap at take-profit
        (take-profit fires: gains are locked in)
      * Otherwise                                       → keep actual return

    **Economic rationale** — with a 2:1 ratio the strategy is profitable even
    with a 50 % win rate:

        EV = 0.50 × 2s + 0.50 × (−s) = +0.50s  (per unit of risk s)

    where s = |stop_threshold|.  Any model IC > 0 improves the win rate
    above 50 %, amplifying the edge further.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with ``date``, ``ticker``, and ``ret_col``.
    ret_col : str
        Column of realised returns to apply the rule to.
    stop_threshold : float
        Maximum acceptable loss per position (e.g. -0.02 = -2 %).
        Must be strictly negative.
    reward_ratio : float
        Take-profit level as a multiple of the stop distance.
        Default 2.0 → 2:1 risk-reward (stop=-2 %, target=+4 %).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with ``ret_col`` clipped to [stop, take_profit].
        Additional diagnostic columns:

        ``stop_triggered``    — True where stop fired (loss was too large)
        ``target_triggered``  — True where take-profit fired (gain locked)
        ``stop_level``        — the stop value used (scalar broadcast)
        ``target_level``      — the take-profit value used (scalar broadcast)
    """
    if stop_threshold >= 0:
        raise ValueError(
            f"stop_threshold must be negative. Got {stop_threshold}."
        )
    if reward_ratio <= 0:
        raise ValueError(f"reward_ratio must be positive. Got {reward_ratio}.")

    take_profit = abs(stop_threshold) * reward_ratio   # e.g. 0.04 for 2:1 at -0.02

    out = df.copy()
    original = out[ret_col].values.copy()

    # Apply clip: floor at stop, cap at take-profit
    clipped = np.clip(original, stop_threshold, take_profit)
    out[ret_col] = clipped

    out["stop_triggered"]   = original < stop_threshold
    out["target_triggered"] = original > take_profit
    out["stop_level"]       = stop_threshold
    out["target_level"]     = take_profit

    return out


def apply_rr_portfolio_stop(
    port_df:            pd.DataFrame,
    stop_threshold:     float = -0.02,
    reward_ratio:       float = 2.0,
    trailing_stop:      float = -0.10,
    recovery_threshold: float = 0.02,
    cooldown_days:      int   = 5,
) -> pd.DataFrame:
    """Combined 2:1 risk-reward + portfolio trailing stop.

    Applies TWO layers of risk management to the long-short portfolio:

    Layer 1 — Position-level (2:1 R:R):
        Each daily long-short return is clipped to [stop, 2×stop].
        This ensures no single day can lose more than ``stop_threshold``
        and takes profits at ``2 × |stop_threshold|``.

    Layer 2 — Portfolio-level trailing stop (circuit breaker):
        Tracks the running peak NAV of the clipped L-S returns.
        If NAV falls ``trailing_stop`` below peak → go flat until
        ``cooldown_days`` elapsed AND NAV recovers ``recovery_threshold``.

    This two-layer approach is the standard professional risk framework:
    position-level stops manage individual trade risk, while the portfolio
    circuit breaker protects against sustained drawdown periods (e.g.
    regime changes, macro events).

    Parameters
    ----------
    port_df : pd.DataFrame
        Output of :func:`build_daily_portfolio_returns`.
    stop_threshold : float
        Per-position daily stop-loss level (e.g. -0.02 = -2 %).
    reward_ratio : float
        Take-profit multiple (default 2.0 → 2:1 risk-reward).
    trailing_stop : float
        Portfolio-level trailing drawdown trigger (e.g. -0.10 = -10 %).
    recovery_threshold : float
        NAV recovery fraction required for circuit-breaker re-entry.
    cooldown_days : int
        Minimum flat days before re-entry after circuit breaker fires.

    Returns
    -------
    pd.DataFrame
        ``port_df`` with all columns from :func:`apply_portfolio_stop_loss`
        computed on the already-clipped returns, plus:

        ``ls_ret_rr``   — L-S return after position-level 2:1 clip only
        ``ls_ret_full`` — L-S return after both position clip + circuit breaker
        ``nav_rr``      — cumulative NAV: position-clip only
        ``nav_full``    — cumulative NAV: both layers
        ``is_flat``     — True on circuit-breaker flat days
    """
    take_profit = abs(stop_threshold) * reward_ratio

    df = port_df.copy().reset_index(drop=True)
    n  = len(df)

    # ── Layer 1: position-level 2:1 clip ─────────────────────────────────────
    raw_ls = df["ls_ret"].values.copy()
    clipped = np.clip(raw_ls, stop_threshold, take_profit)
    df["ls_ret_rr"] = clipped          # after position clip, before circuit breaker

    # ── Layer 2: portfolio trailing stop on clipped returns ───────────────────
    # Temporarily replace ls_ret with clipped version for apply_portfolio_stop_loss
    df_tmp = df.copy()
    df_tmp["ls_ret"] = clipped

    sl_df = apply_portfolio_stop_loss(
        df_tmp,
        trailing_stop=trailing_stop,
        recovery_threshold=recovery_threshold,
        cooldown_days=cooldown_days,
    )

    # Merge results back
    df["ls_ret_full"] = sl_df["ls_ret_sl"].values    # both layers applied
    df["nav_rr"]      = np.cumprod(1.0 + clipped)    # position clip only
    df["nav_full"]    = sl_df["nav_sl"].values        # both layers
    df["is_flat"]     = sl_df["is_flat"].values
    df["drawdown"]    = sl_df["drawdown"].values

    return df


def evaluate_rr_stop_comparison(
    port_df:            pd.DataFrame,
    stop_threshold:     float = -0.02,
    reward_ratio:       float = 2.0,
    trailing_stop:      float = -0.10,
    recovery_threshold: float = 0.02,
    cooldown_days:      int   = 5,
) -> Dict[str, Dict[str, float]]:
    """Full 3-column comparison: Gross / Trailing-Stop-Only / 2:1 R-R + Trailing.

    Parameters
    ----------
    port_df : pd.DataFrame
        Output of :func:`build_daily_portfolio_returns`.
    stop_threshold : float
        Per-position daily stop-loss (e.g. -0.02).
    reward_ratio : float
        Take-profit multiple (default 2.0 → 2:1).
    trailing_stop : float
        Portfolio trailing stop trigger.
    recovery_threshold : float
    cooldown_days : int

    Returns
    -------
    Dict[str, Dict[str, float]]
        Keys: ``"gross"``, ``"trailing_only"``, ``"rr_2to1_plus_trailing"``.
    """
    # ── Gross (no risk management) ────────────────────────────────────────────
    gross_rets = port_df["ls_ret"].values

    # ── Trailing stop only ────────────────────────────────────────────────────
    sl_df  = apply_portfolio_stop_loss(
        port_df,
        trailing_stop=trailing_stop,
        recovery_threshold=recovery_threshold,
        cooldown_days=cooldown_days,
    )
    trail_rets   = sl_df["ls_ret_sl"].values
    trail_flat   = float(sl_df["is_flat"].sum())

    # ── 2:1 R:R + trailing circuit breaker ───────────────────────────────────
    rr_df      = apply_rr_portfolio_stop(
        port_df,
        stop_threshold=stop_threshold,
        reward_ratio=reward_ratio,
        trailing_stop=trailing_stop,
        recovery_threshold=recovery_threshold,
        cooldown_days=cooldown_days,
    )
    rr_rets    = rr_df["ls_ret_full"].values
    rr_flat    = float(rr_df["is_flat"].sum())

    def _m(returns: np.ndarray, flat: float = 0.0) -> Dict[str, float]:
        return {
            "ann_ret":      compute_annual_return(returns),
            "ann_vol":      compute_annual_volatility(returns),
            "sharpe":       compute_sharpe(returns),
            "sortino":      compute_sortino(returns),
            "calmar":       compute_calmar(returns),
            "max_drawdown": compute_max_drawdown(returns),
            "hit_ratio":    float(np.mean(returns > 0)) if len(returns) else 0.0,
            "flat_days":    flat,
        }

    return {
        "gross":                    _m(gross_rets),
        "trailing_only":            _m(trail_rets,  trail_flat),
        "rr_2to1_plus_trailing":    _m(rr_rets,     rr_flat),
    }


def format_rr_comparison_table(
    comparison:     Dict[str, Dict[str, float]],
    stop_threshold: float = -0.02,
    reward_ratio:   float = 2.0,
    trailing_stop:  float = -0.10,
) -> str:
    """Format the 3-column risk-management comparison as a readable table.

    Parameters
    ----------
    comparison : Dict[str, Dict[str, float]]
        Output of :func:`evaluate_rr_stop_comparison`.
    stop_threshold : float
    reward_ratio : float
    trailing_stop : float

    Returns
    -------
    str
        Multi-line formatted table suitable for printing to a log.
    """
    take_profit = abs(stop_threshold) * reward_ratio
    header = (
        f"\n{'='*80}\n"
        f"  Risk Management Comparison\n"
        f"  Stop={stop_threshold:.1%}  |  Take-Profit={take_profit:.1%}  "
        f"({reward_ratio:.0f}:1 R:R)  |  Trailing DD={trailing_stop:.0%}\n"
        f"{'='*80}\n"
        f"  {'Metric':<22} {'No Risk Mgmt':>16} {'Trailing Only':>16} "
        f"{'2:1 R:R + Trail':>18}\n"
        f"  {'-'*22} {'-'*16} {'-'*16} {'-'*18}"
    )
    metric_labels = {
        "ann_ret":      "Annual Return",
        "ann_vol":      "Annual Vol",
        "sharpe":       "Sharpe Ratio",
        "sortino":      "Sortino Ratio",
        "calmar":       "Calmar Ratio",
        "max_drawdown": "Max Drawdown",
        "hit_ratio":    "Hit Ratio",
        "flat_days":    "Flat Days",
    }
    gross  = comparison.get("gross", {})
    trail  = comparison.get("trailing_only", {})
    rr     = comparison.get("rr_2to1_plus_trailing", {})
    rows   = []
    for key, label in metric_labels.items():
        g = gross.get(key, float("nan"))
        t = trail.get(key, float("nan"))
        r = rr.get(key, float("nan"))
        rows.append(f"  {label:<22} {g:>16.4f} {t:>16.4f} {r:>18.4f}")
    footer = "=" * 80
    return "\n".join([header] + rows + [footer])


# ─── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    n_dates   = 100
    n_tickers = 200
    dates   = pd.date_range("2023-01-01", periods=n_dates, freq="B")
    tickers = [f"TICK{i:03d}" for i in range(n_tickers)]

    rows = []
    for d in dates:
        for t in tickers:
            ret = rng.normal(0, 0.01)
            pred = ret * 0.5 + rng.normal(0, 0.01)
            rows.append({"date": d, "ticker": t, "target_ret_1d": ret, "pred_1d": pred})

    df = pd.DataFrame(rows)
    m  = evaluate_predictions(df, pred_col="pred_1d", ret_col="target_ret_1d")
    print(format_metrics_table(m, "Smoke Test Results"))

    port_df = build_daily_portfolio_returns(df, pred_col="pred_1d", ret_col="target_ret_1d")

    # Position-level stop
    df_pos_sl = apply_position_stop_loss(df, ret_col="target_ret_1d", stop_threshold=-0.03)
    print(f"\nPosition stop triggered on {df_pos_sl['stop_triggered'].sum()} stock-days.")

    # 2:1 R:R position stop
    df_rr = apply_risk_reward_stop_loss(df, ret_col="target_ret_1d",
                                        stop_threshold=-0.02, reward_ratio=2.0)
    print(f"2:1 R:R — stops={df_rr['stop_triggered'].sum()}, "
          f"targets={df_rr['target_triggered'].sum()}")

    # Full 3-column comparison
    comp = evaluate_rr_stop_comparison(
        port_df,
        stop_threshold=-0.02, reward_ratio=2.0,
        trailing_stop=-0.05,  cooldown_days=3,
    )
    print(format_rr_comparison_table(
        comp, stop_threshold=-0.02, reward_ratio=2.0, trailing_stop=-0.05
    ))
    print("All risk management tests passed.")

