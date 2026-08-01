import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from .utils import setup_logger, export_csv_table, export_latex_table
import matplotlib.pyplot as plt

logger = setup_logger("BacktestEngine")

class BacktestEngine:
    """
    Vectorized portfolio backtesting engine.
    Constructs Long/Short decile portfolios based on ML predictions.
    Incorporates daily turnover-adjusted transaction costs and realistic constraints.
    """
    def __init__(self, data_dir: Path, results_dir: Path, figures_dir: Path, feature_size='11'):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.figures_dir = figures_dir
        self.tables_dir = self.results_dir / "tables"
        self.feature_size = feature_size
        self.rf_daily = 0.0525 / 252  # Default fallback daily risk-free rate

    def _compute_drawdown(self, returns: pd.Series):
        """Computes max drawdown and max drawdown duration."""
        cum = (1 + returns).cumprod()
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max
        
        in_drawdown = cum < running_max
        drawdown_periods = in_drawdown.astype(int).groupby((~in_drawdown).cumsum()).sum()
        max_duration = drawdown_periods.max() if not drawdown_periods.empty else 0
        
        return dd.min(), max_duration

    def _compute_metrics(self, returns: pd.Series, tcost_bps: float, turnover: pd.Series = None, is_ls: bool = False):
        """Computes comprehensive risk-adjusted metrics, subtracting turnover-adjusted costs daily."""
        if turnover is not None:
            # One-way cost is tcost_bps / 10000.
            # Rebalancing requires selling departed assets and buying new assets.
            # Daily cost = 2 * turnover * (tcost_bps / 10000).
            trade_cost = 2.0 * turnover * (tcost_bps / 10000.0)
            s = returns - trade_cost
        else:
            s = returns
            
        # Subtract risk-free rate for long-only portfolios; long-short is self-financing
        rf_adjust = 0.0 if is_ls else self.rf_daily
        
        ann_ret = (s.mean() - rf_adjust) * 252
        ann_vol = s.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        
        # Sortino Ratio (Downside deviation)
        downside = s[s < rf_adjust] - rf_adjust
        ann_down_vol = downside.std() * np.sqrt(252)
        sortino = ann_ret / ann_down_vol if (len(downside) > 0 and ann_down_vol > 0) else 0.0
        
        max_dd, max_dd_dur = self._compute_drawdown(s - rf_adjust)
        
        # 95% Daily Historical Value at Risk (VaR) and Expected Shortfall (ES)
        var_95 = np.percentile(s - rf_adjust, 5) if len(s) > 0 else 0.0
        es_95 = (s - rf_adjust)[(s - rf_adjust) <= var_95].mean() if len(s) > 0 else 0.0
        
        return {
            'Ann. Excess Return': ann_ret,
            'Ann. Volatility': ann_vol,
            'Sharpe': sharpe,
            'Sortino': sortino,
            'Max Drawdown': max_dd,
            'Max DD Duration (Days)': max_dd_dur,
            'VaR 95% Daily': var_95,
            'ES 95% Daily': es_95
        }

    def run_backtest(self, horizon='1d'):
        """Executes the daily decile spread backtest."""
        logger.info(f"Loading OOS predictions for backtesting (Horizon: {horizon}, Feature Size: {self.feature_size})...")
        oos_path = self.data_dir / "processed" / f"oos_predictions_{horizon}_feat{self.feature_size}.parquet"
        if not oos_path.exists():
            oos_path = self.data_dir / "processed" / f"oos_predictions_{horizon}.parquet"
            if not oos_path.exists():
                oos_path = self.data_dir / "processed" / "oos_predictions.parquet"
                if not oos_path.exists():
                    logger.error(f"OOS predictions not found for feature size {self.feature_size} and horizon {horizon}.")
                    return
            
        df = pd.read_parquet(oos_path)
        if 'mkt_ret' not in df.columns and 'mkt_rf' in df.columns and 'rf' in df.columns:
            df['mkt_ret'] = df['mkt_rf'] + df['rf']
        df = df.sort_values(['ticker', 'date'])
        
        # Shift predictions per ticker by 1 day to ensure lookahead-free trading
        pred_cols = {
            'LASSO': 'pred_prob_lasso',
            'ElasticNet': 'pred_prob_elasticnet',
            'RF': 'pred_prob_rf',
            'XGB': 'pred_prob_xgb',
            'LSTM': 'pred_prob_lstm'
        }
        # SIGNAL-RETURN ALIGNMENT FIX (Audit Fix NEW-C1)
        # ================================================
        # The C2 fix shifted targets to shift(-2), meaning:
        #   prediction at Close(t) → predicts return at Close(t+1)→Close(t+2)
        # Therefore in the backtest, the prediction made at time t must be
        # applied to the return observed at day t+2.  shift(2) on predictions
        # means: on day t's row, we use the prediction from day t-2, which
        # was targeting the return at day (t-2)+2 = t.  ✓
        for name, col in pred_cols.items():
            if col in df.columns:
                df[f"{col}_shifted"] = df.groupby('ticker')[col].shift(2)
                
        # Filter out delisted stocks by checking volume & return standard deviation
        vol_col = 'px_volume' if 'px_volume' in df.columns else 'volume'
        df['vol_std_5d'] = df.groupby('ticker')[vol_col].transform(lambda x: x.rolling(5).std())
        df['ret_std_5d'] = df.groupby('ticker')['ret'].transform(lambda x: x.rolling(5).std())
        df['vol_std_5d_shifted'] = df.groupby('ticker')['vol_std_5d'].shift(2)
        df['ret_std_5d_shifted'] = df.groupby('ticker')['ret_std_5d'].shift(2)
        
        df = df[
            (df['vol_std_5d_shifted'] > 0) &
            (df['ret_std_5d_shifted'] > 0)
        ].dropna(subset=[f"{pred_cols[m]}_shifted" for m in pred_cols if pred_cols[m] in df.columns])
        
        unique_dates = sorted(df['date'].unique())
        
        # Dynamically set average risk-free rate from the test period
        if 'rf' in df.columns:
            self.rf_daily = df['rf'].mean()
            logger.info(f"Average daily risk-free rate in test set: {self.rf_daily:.6f} ({self.rf_daily*252*100:.2f}% annualized)")
            
        models = ['LASSO', 'ElasticNet', 'RF', 'XGB', 'LSTM']
        
        prev_longs = {m: set() for m in models}
        prev_shorts = {m: set() for m in models}
        
        daily_records = []
        for d in unique_dates:
            day_stocks = df[df['date'] == d]
            n_stocks = len(day_stocks)
            if n_stocks < 50:
                continue
                
            cutoff = int(np.ceil(n_stocks * 0.10))
            sp_ret = day_stocks['mkt_ret'].iloc[0] if not day_stocks.empty else 0.0
            
            row = {'date': d, 'SP500': sp_ret}
            for fcol in ['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'rf']:
                if fcol in day_stocks.columns:
                    row[fcol] = day_stocks[fcol].iloc[0]
            
            for name in models:
                pred_col = f"{pred_cols[name]}_shifted"
                if pred_col not in day_stocks.columns:
                    continue
                day_sorted = day_stocks.sort_values(pred_col, ascending=False)
                
                long_set = set(day_sorted.iloc[:cutoff]['ticker'])
                short_set = set(day_sorted.iloc[-cutoff:]['ticker'])
                
                long_ret = day_sorted.iloc[:cutoff]['ret'].mean()
                short_ret = day_sorted.iloc[-cutoff:]['ret'].mean()
                
                # TURNOVER CALCULATION FIX (Audit Fix M9)
                # ==========================================
                # Improved turnover to account for weight drift, not just
                # name-based membership changes. Even if all stocks remain,
                # differential returns cause weights to deviate from equal-weight,
                # requiring rebalancing (and transaction costs).
                #
                # turnover = fraction_new_names + overlap_rebalancing_cost
                # where overlap_rebalancing_cost = overlap_fraction * (1/n_stocks)
                # because equal-weight portfolios need full rebalancing.
                #
                # For equal-weight portfolios, the correct one-way turnover is:
                # turnover = 1 - (n_retained / n_total)  [name-based, lower bound]
                # This is a lower bound because retained stocks also need
                # rebalancing from drifted weights back to 1/n.
                if prev_longs[name]:
                    n_retained_long = len(long_set & prev_longs[name])
                    # Name-based turnover + rebalancing adjustment
                    # Stocks that stay still need ~(1/cutoff) rebalancing on average
                    name_turnover_long = 1.0 - (n_retained_long / len(long_set))
                    rebalance_cost_long = (n_retained_long / len(long_set)) * (1.0 / cutoff)
                    turnover_long = name_turnover_long + rebalance_cost_long
                else:
                    turnover_long = 1.0
                    
                if prev_shorts[name]:
                    n_retained_short = len(short_set & prev_shorts[name])
                    name_turnover_short = 1.0 - (n_retained_short / len(short_set))
                    rebalance_cost_short = (n_retained_short / len(short_set)) * (1.0 / cutoff)
                    turnover_short = name_turnover_short + rebalance_cost_short
                else:
                    turnover_short = 1.0
                    
                row[f'{name}_Long'] = long_ret
                # Short portfolio return represents holding the bottom decile long
                row[f'{name}_Short'] = short_ret
                # Long-Short portfolio return: Long_Ret - Short_Ret
                row[f'{name}_LS'] = long_ret - short_ret
                
                # TAKE-PROFIT / STOP-LOSS FIX (Audit Fix C6)
                # =============================================
                # Implement the stock-level daily TP/SL exit rule claimed in
                # the thesis methodology (2:1 ratio: +4% TP / -2% SL).
                # This clips individual stock returns BEFORE portfolio aggregation.
                #
                # IMPORTANT DISCLAIMER: This assumes perfect intraday limit-order
                # execution at the TP/SL price levels. In reality, overnight gaps
                # and intraday volatility mean exact execution is not guaranteed.
                # Results with TPSL should be interpreted as an UPPER BOUND on
                # the risk management benefit. See Thesis Section 5.3.
                tp_limit = 0.04   # Take-profit: cap gains at +4%
                sl_limit = -0.02  # Stop-loss: cap losses at -2%
                
                long_rets_clipped = day_sorted.iloc[:cutoff]['ret'].clip(lower=sl_limit, upper=tp_limit)
                short_rets_clipped = day_sorted.iloc[-cutoff:]['ret'].clip(lower=sl_limit, upper=tp_limit)
                
                row[f'{name}_Long_TPSL'] = long_rets_clipped.mean()
                row[f'{name}_Short_TPSL'] = short_rets_clipped.mean()
                row[f'{name}_LS_TPSL'] = long_rets_clipped.mean() - short_rets_clipped.mean()
                
                row[f'{name}_Long_TO'] = turnover_long
                row[f'{name}_Short_TO'] = turnover_short
                
                prev_longs[name] = long_set
                prev_shorts[name] = short_set
                
            daily_records.append(row)
            
        port_df = pd.DataFrame(daily_records).set_index('date')
        
        # --- Stop-Loss / Drawdown Circuit Breakers ---
        logger.info("Computing Drawdown Circuit Breaker Stop-Loss Scenarios (-15% stop-out)...")
        for name in models:
            for suffix in ['Long', 'Short', 'LS']:
                col = f"{name}_{suffix}"
                if col not in port_df.columns:
                    continue
                rets = port_df[col].copy()
                to_col = f"{name}_Long_TO" if suffix == 'Long' else (f"{name}_Short_TO" if suffix == 'Short' else None)
                if suffix == 'LS':
                    to_val = port_df[f"{name}_Long_TO"] + port_df[f"{name}_Short_TO"]
                else:
                    to_val = port_df[to_col] if to_col in port_df.columns else None
                
                if to_val is not None:
                    net_rets = rets - 2.0 * to_val * (10 / 10000.0)
                else:
                    net_rets = rets
                
                cum_wealth = (1 + net_rets).cumprod()
                running_max = cum_wealth.cummax()
                drawdown = (cum_wealth - running_max) / running_max
                
                stop_out_mask = drawdown <= -0.15
                if stop_out_mask.any():
                    stop_date = stop_out_mask.idxmax()
                    rets.loc[stop_date:] = 0.0 if suffix == 'LS' else self.rf_daily
                    logger.info(f"  [Stop-Loss] Strategy {col} triggered stop-out on {stop_date.strftime('%Y-%m-%d')} after exceeding -15% drawdown.")
                
                port_df[f"{col}_StopLoss"] = rets
        
        logger.info("Computing Transaction Cost Scenarios (0, 5, 10, 20 bps)...")
        rows = []
        for cost_bps in [0, 5, 10, 20]:
            # SP500 is buy-and-hold (no cost)
            m_sp = self._compute_metrics(port_df['SP500'], tcost_bps=0, turnover=None, is_ls=False)
            m_sp['Strategy'] = 'SP500'
            m_sp['TxCost (bps)'] = cost_bps
            rows.append(m_sp)
            
            for name in models:
                if f'{name}_Long' not in port_df.columns:
                    continue
                to_long = port_df[f'{name}_Long_TO']
                m_long = self._compute_metrics(port_df[f'{name}_Long'], tcost_bps=cost_bps, turnover=to_long, is_ls=False)
                m_long['Strategy'] = f'{name}_Long'
                m_long['TxCost (bps)'] = cost_bps
                rows.append(m_long)
                
                to_short = port_df[f'{name}_Short_TO']
                m_short = self._compute_metrics(port_df[f'{name}_Short'], tcost_bps=cost_bps, turnover=to_short, is_ls=False)
                m_short['Strategy'] = f'{name}_Short'
                m_short['TxCost (bps)'] = cost_bps
                rows.append(m_short)
                
                to_ls = to_long + to_short
                m_ls = self._compute_metrics(port_df[f'{name}_LS'], tcost_bps=cost_bps, turnover=to_ls, is_ls=True)
                m_ls['Strategy'] = f'{name}_LS'
                m_ls['TxCost (bps)'] = cost_bps
                rows.append(m_ls)
                
                # Append StopLoss metrics
                if f'{name}_LS_StopLoss' in port_df.columns:
                    m_ls_sl = self._compute_metrics(port_df[f'{name}_LS_StopLoss'], tcost_bps=cost_bps, turnover=to_ls, is_ls=True)
                    m_ls_sl['Strategy'] = f'{name}_LS_StopLoss'
                    m_ls_sl['TxCost (bps)'] = cost_bps
                    rows.append(m_ls_sl)
                
                # Append TPSL metrics (Audit Fix C6)
                if f'{name}_Long_TPSL' in port_df.columns:
                    m_long_tpsl = self._compute_metrics(port_df[f'{name}_Long_TPSL'], tcost_bps=cost_bps, turnover=to_long, is_ls=False)
                    m_long_tpsl['Strategy'] = f'{name}_Long_TPSL'
                    m_long_tpsl['TxCost (bps)'] = cost_bps
                    rows.append(m_long_tpsl)
                    
                    m_ls_tpsl = self._compute_metrics(port_df[f'{name}_LS_TPSL'], tcost_bps=cost_bps, turnover=to_ls, is_ls=True)
                    m_ls_tpsl['Strategy'] = f'{name}_LS_TPSL'
                    m_ls_tpsl['TxCost (bps)'] = cost_bps
                    rows.append(m_ls_tpsl)
                    
        metrics_df = pd.DataFrame(rows).set_index(['Strategy', 'TxCost (bps)'])
        
        # Log summary table for 10 bps scenario
        logger.info(f"\n--- Performance Summary under 10 bps Transaction Cost ({horizon}) ---")
        logger.info("\n" + metrics_df.loc[(slice(None), 10), :][['Ann. Excess Return', 'Sharpe', 'Max Drawdown', 'VaR 95% Daily']].to_string())
        
        # Export tables
        export_csv_table(metrics_df, self.tables_dir / f"backtest_metrics_{horizon}_feat{self.feature_size}.csv")
        export_latex_table(metrics_df.loc[(slice(None), 10), :], self.tables_dir / f"backtest_metrics_10bps_{horizon}_feat{self.feature_size}.tex")
        export_latex_table(metrics_df, self.tables_dir / f"backtest_metrics_grid_{horizon}_feat{self.feature_size}.tex")
        
        # Backward compatibility / default feature size exports
        if self.feature_size == '11':
            export_csv_table(metrics_df, self.tables_dir / f"backtest_metrics_{horizon}.csv")
            export_latex_table(metrics_df.loc[(slice(None), 10), :], self.tables_dir / f"backtest_metrics_10bps_{horizon}.tex")
            export_latex_table(metrics_df, self.tables_dir / f"backtest_metrics_grid_{horizon}.tex")
            if horizon == '1d':
                export_csv_table(metrics_df, self.tables_dir / "backtest_metrics.csv")
                export_latex_table(metrics_df.loc[(slice(None), 10), :], self.tables_dir / "backtest_metrics_10bps.tex")
                export_latex_table(metrics_df, self.tables_dir / "backtest_metrics_grid.tex")
            
        logger.info(f"Saved backtesting LaTeX and CSV tables for horizon {horizon}.")
        
        self._plot_equity_curves(port_df, cost_bps=10, horizon=horizon)
        if 'XGB_Short' in port_df.columns:
            self._plot_rolling_sharpe(port_df['XGB_Short'] - 2.0 * port_df['XGB_Short_TO'] * (10 / 10000.0), horizon=horizon)
            
        # --- Run OOS Regime Analysis for this horizon ---
        self._run_regime_analysis(port_df, horizon=horizon)
        
        # --- Run Historical Stress Scenario Analysis ---
        self._run_historical_stress_tests(port_df, horizon=horizon)
        
        # --- Run Fama-French Factor Spanning Regressions (Audit Fix NEW-M2) ---
        self._run_fama_french_spanning_regressions(port_df, horizon=horizon)

    def _run_fama_french_spanning_regressions(self, port_df: pd.DataFrame, horizon: str, cost_bps: float = 10):
        """Runs Fama-French 5-factor spanning regressions on net Long-Short portfolio returns (Audit Fix NEW-M2)."""
        logger.info(f"\n--- Running Fama-French Factor Spanning Regressions ({horizon} Horizon) ---")
        factors = ['mkt_rf', 'smb', 'hml', 'rmw', 'cma']
        avail_factors = [f for f in factors if f in port_df.columns]
        
        if not avail_factors:
            logger.warning("Fama-French factor columns not found in port_df. Skipping spanning regressions.")
            return
            
        models = ['LASSO', 'ElasticNet', 'RF', 'XGB', 'LSTM']
        spanning_rows = []
        
        for name in models:
            ls_col = f'{name}_LS'
            to_col_long = f'{name}_Long_TO'
            to_col_short = f'{name}_Short_TO'
            
            if ls_col not in port_df.columns:
                continue
                
            # Compute net Long-Short return after transaction costs
            to_total = port_df[to_col_long] + port_df[to_col_short] if (to_col_long in port_df.columns and to_col_short in port_df.columns) else 0.0
            net_ls_ret = port_df[ls_col] - 2.0 * to_total * (cost_bps / 10000.0)
            
            # Align with available factors
            reg_df = pd.DataFrame({'ret': net_ls_ret}).join(port_df[avail_factors]).dropna()
            if len(reg_df) < 50:
                continue
                
            Y = reg_df['ret']
            X = sm.add_constant(reg_df[avail_factors])
            
            # OLS with Newey-West HAC standard errors (maxlags=5 for daily data)
            try:
                model_fit = sm.OLS(Y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
                
                row_dict = {
                    'Strategy': f'{name}_LS_net{int(cost_bps)}bps',
                    'Alpha (Ann. %)': model_fit.params['const'] * 252 * 100,
                    'Alpha t-stat': model_fit.tvalues['const'],
                    'R2': model_fit.rsquared
                }
                for f in avail_factors:
                    row_dict[f'Beta_{f}'] = model_fit.params[f]
                    row_dict[f't_{f}'] = model_fit.tvalues[f]
                    
                spanning_rows.append(row_dict)
            except Exception as e:
                logger.warning(f"Error running FF regression for {name}: {e}")
            
        if spanning_rows:
            ff_df = pd.DataFrame(spanning_rows).set_index('Strategy')
            logger.info("\n" + ff_df.to_string())
            export_csv_table(ff_df, self.tables_dir / f"ff_spanning_regressions_{horizon}_feat{self.feature_size}.csv")
            export_latex_table(ff_df, self.tables_dir / f"ff_spanning_regressions_{horizon}_feat{self.feature_size}.tex")
            if self.feature_size == '11' and horizon == '1d':
                export_csv_table(ff_df, self.tables_dir / "ff_spanning_regressions.csv")
                export_latex_table(ff_df, self.tables_dir / "ff_spanning_regressions.tex")

    def _run_historical_stress_tests(self, port_df: pd.DataFrame, horizon: str, cost_bps: float = 10):
        """Runs stress tests for three specific macroeconomic crisis sub-periods."""
        logger.info(f"\n--- Running Historical Stress Scenario Analysis ({horizon} Horizon) ---")
        
        scenarios = {
            'COVID-19 Crash (Feb-Apr 2020)': ('2020-02-20', '2020-04-30'),
            'Rate Hike Sell-Off (2022)': ('2022-01-01', '2022-10-31'),
            'Yen Carry Trade Panic (Aug 2024)': ('2024-07-25', '2024-08-15')
        }
        
        models = ['LASSO', 'ElasticNet', 'RF', 'XGB', 'LSTM']
        stress_rows = []
        
        for name, (start_dt, end_dt) in scenarios.items():
            sub_df = port_df.loc[start_dt:end_dt]
            if sub_df.empty:
                continue
                
            # SP500
            m_sp = self._compute_metrics(sub_df['SP500'], tcost_bps=0, turnover=None, is_ls=False)
            m_sp['Strategy'] = 'SP500'
            m_sp['Scenario'] = name
            stress_rows.append(m_sp)
            
            for m in models:
                col = f"{m}_LS"
                if col not in sub_df.columns:
                    continue
                to_val = sub_df[f"{m}_Long_TO"] + sub_df[f"{m}_Short_TO"]
                m_ls = self._compute_metrics(sub_df[col], tcost_bps=cost_bps, turnover=to_val, is_ls=True)
                m_ls['Strategy'] = col
                m_ls['Scenario'] = name
                stress_rows.append(m_ls)
                
                # Add the stop-loss version to compare
                sl_col = f"{col}_StopLoss"
                if sl_col in sub_df.columns:
                    m_sl = self._compute_metrics(sub_df[sl_col], tcost_bps=cost_bps, turnover=to_val, is_ls=True)
                    m_sl['Strategy'] = f"{col}_StopLoss"
                    m_sl['Scenario'] = name
                    stress_rows.append(m_sl)
                    
        stress_df = pd.DataFrame(stress_rows).set_index(['Strategy', 'Scenario'])
        logger.info("\n" + stress_df[['Ann. Excess Return', 'Sharpe', 'Max Drawdown']].to_string())
        
        export_csv_table(stress_df, self.tables_dir / f"backtest_stress_{horizon}_feat{self.feature_size}.csv")
        export_latex_table(stress_df, self.tables_dir / f"backtest_stress_{horizon}_feat{self.feature_size}.tex")
        if self.feature_size == '11':
            export_csv_table(stress_df, self.tables_dir / f"backtest_stress_{horizon}.csv")
            export_latex_table(stress_df, self.tables_dir / f"backtest_stress_{horizon}.tex")
            if horizon == '1d':
                export_csv_table(stress_df, self.tables_dir / "backtest_stress.csv")
                export_latex_table(stress_df, self.tables_dir / "backtest_stress.tex")
        logger.info(f"Saved historical stress testing LaTeX and CSV tables for horizon {horizon}.")

    def _run_regime_analysis(self, port_df: pd.DataFrame, horizon: str, cost_bps: float = 10):
        """Runs out-of-sample sub-period analysis: COVID Pandemic years (2020-2021) vs. Post-COVID years (2022-2024)."""
        logger.info(f"\n--- Running Out-of-Sample Sub-Period Regime Analysis ({horizon} Horizon) ---")
        
        regimes = {
            'COVID_Pandemic_OOS (2020-2021)': ('2020-01-01', '2021-12-31'),
            'Post_COVID_OOS (2022-2024)': ('2022-01-01', '2024-12-31')
        }
        
        models = ['LASSO', 'ElasticNet', 'RF', 'XGB', 'LSTM']
        regime_rows = []
        
        for regime_name, (start_dt, end_dt) in regimes.items():
            sub_df = port_df.loc[start_dt:end_dt]
            if sub_df.empty:
                continue
                
            # Compute SP500 (no cost)
            m_sp = self._compute_metrics(sub_df['SP500'], tcost_bps=0, turnover=None, is_ls=False)
            m_sp['Strategy'] = 'SP500'
            m_sp['Regime'] = regime_name
            regime_rows.append(m_sp)
            
            # Compute ML portfolios
            for name in models:
                if f'{name}_Long' not in sub_df.columns:
                    continue
                to_long = sub_df[f'{name}_Long_TO']
                m_long = self._compute_metrics(sub_df[f'{name}_Long'], tcost_bps=cost_bps, turnover=to_long, is_ls=False)
                m_long['Strategy'] = f'{name}_Long'
                m_long['Regime'] = regime_name
                regime_rows.append(m_long)
                
                to_short = sub_df[f'{name}_Short_TO']
                # Short portfolio represents holding the bottom decile long (is_ls=False)
                m_short = self._compute_metrics(sub_df[f'{name}_Short'], tcost_bps=cost_bps, turnover=to_short, is_ls=False)
                m_short['Strategy'] = f'{name}_Short'
                m_short['Regime'] = regime_name
                regime_rows.append(m_short)
                
                to_ls = to_long + to_short
                m_ls = self._compute_metrics(sub_df[f'{name}_LS'], tcost_bps=cost_bps, turnover=to_ls, is_ls=True)
                m_ls['Strategy'] = f'{name}_LS'
                m_ls['Regime'] = regime_name
                regime_rows.append(m_ls)
                
        regime_metrics_df = pd.DataFrame(regime_rows).set_index(['Strategy', 'Regime'])
        logger.info("\n" + regime_metrics_df[['Ann. Excess Return', 'Sharpe', 'Max Drawdown']].to_string())
        
        export_csv_table(regime_metrics_df, self.tables_dir / f"backtest_metrics_regimes_{horizon}_feat{self.feature_size}.csv")
        export_latex_table(regime_metrics_df, self.tables_dir / f"backtest_metrics_regimes_{horizon}_feat{self.feature_size}.tex")
        if self.feature_size == '11':
            export_csv_table(regime_metrics_df, self.tables_dir / f"backtest_metrics_regimes_{horizon}.csv")
            export_latex_table(regime_metrics_df, self.tables_dir / f"backtest_metrics_regimes_{horizon}.tex")
            if horizon == '1d':
                export_csv_table(regime_metrics_df, self.tables_dir / "backtest_metrics_regimes.csv")
                export_latex_table(regime_metrics_df, self.tables_dir / "backtest_metrics_regimes.tex")
        logger.info(f"Saved regime analysis LaTeX and CSV tables for horizon {horizon}.")

    def _plot_equity_curves(self, port_df: pd.DataFrame, cost_bps: float = 10, horizon='1d'):
        """Plots cumulative wealth over the OOS period net of turnover-adjusted costs."""
        net_returns = pd.DataFrame(index=port_df.index)
        net_returns['SP500'] = port_df['SP500'] - self.rf_daily
        
        models = ['LASSO', 'ElasticNet', 'RF', 'XGB', 'LSTM']
        for name in models:
            if f'{name}_Long' not in port_df.columns:
                continue
            cost_long = 2.0 * port_df[f'{name}_Long_TO'] * (cost_bps / 10000.0)
            net_returns[f'{name}_Long'] = port_df[f'{name}_Long'] - cost_long - self.rf_daily
            
            cost_short = 2.0 * port_df[f'{name}_Short_TO'] * (cost_bps / 10000.0)
            # Short portfolio represents holding bottom decile long (is_ls=False), so we subtract rf_daily!
            net_returns[f'{name}_Short'] = port_df[f'{name}_Short'] - cost_short - self.rf_daily
            
            cost_ls = cost_long + cost_short
            net_returns[f'{name}_LS'] = port_df[f'{name}_LS'] - cost_ls
            
        wealth = (1 + net_returns).cumprod()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(wealth.index, wealth['SP500'], label='S&P 500 Benchmark (Buy-and-Hold)', color='#64748b', ls='--')
        
        colors = {
            'XGB_Short': '#22c55e',
            'RF_Short': '#3b82f6',
            'ElasticNet_Short': '#ff9f1c',
            'LASSO_Short': '#8b5cf6',
            'XGB_Long': '#ef4444'
        }
        
        for name, col_color in colors.items():
            if name in wealth.columns:
                ax.plot(wealth.index, wealth[name], label=f"{name.replace('_', ' ')} (Net)", color=col_color, lw=2 if 'Short' in name and 'XGB' in name else 1.5)
                
        ax.set_title(f"Out-of-Sample Cumulative Wealth ({horizon} Horizon, Net of {cost_bps} bps Rebalancing Cost)", fontweight='bold', pad=15)
        ax.set_ylabel("Cumulative Excess Wealth", fontweight='bold')
        ax.set_xlabel("Date", fontweight='bold')
        ax.legend(loc="upper left", frameon=True)
        
        fig.tight_layout()
        plot_path = self.figures_dir / f"equity_curves_xgb_{horizon}_feat{self.feature_size}.png"
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved equity curves plot to {plot_path}")
        
        if self.feature_size == '11':
            # Save copies with legacy filenames for 11 features
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(wealth.index, wealth['SP500'], label='S&P 500 Benchmark (Buy-and-Hold)', color='#64748b', ls='--')
            for name, col_color in colors.items():
                if name in wealth.columns:
                    ax.plot(wealth.index, wealth[name], label=f"{name.replace('_', ' ')} (Net)", color=col_color, lw=2 if 'Short' in name and 'XGB' in name else 1.5)
            ax.set_title(f"Out-of-Sample Cumulative Wealth ({horizon} Horizon, Net of {cost_bps} bps Rebalancing Cost)", fontweight='bold', pad=15)
            ax.set_ylabel("Cumulative Excess Wealth", fontweight='bold')
            ax.set_xlabel("Date", fontweight='bold')
            ax.legend(loc="upper left", frameon=True)
            plt.tight_layout()
            plt.savefig(self.figures_dir / f"equity_curves_xgb_{horizon}.png", dpi=300)
            plt.close()
            
            if horizon == '1d':
                wealth_fn = self.figures_dir / "equity_curves_xgb.png"
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(wealth.index, wealth['SP500'], label='S&P 500 Benchmark (Buy-and-Hold)', color='#64748b', ls='--')
                for name, col_color in colors.items():
                    if name in wealth.columns:
                        ax.plot(wealth.index, wealth[name], label=f"{name.replace('_', ' ')} (Net)", color=col_color, lw=2 if 'Short' in name and 'XGB' in name else 1.5)
                ax.set_title(f"Out-of-Sample Cumulative Wealth (Net of {cost_bps} bps Rebalancing Cost)", fontweight='bold', pad=15)
                ax.set_ylabel("Cumulative Excess Wealth", fontweight='bold')
                ax.set_xlabel("Date", fontweight='bold')
                ax.legend(loc="upper left", frameon=True)
                plt.tight_layout()
                plt.savefig(wealth_fn, dpi=300)
                plt.close()

    def _plot_rolling_sharpe(self, returns: pd.Series, horizon='1d', window=126):
        """Plots rolling 6-month Sharpe ratio to visualize regime stability."""
        rolling_ret = (returns.rolling(window).mean() - self.rf_daily) * 252
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_ret / rolling_vol
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(rolling_sharpe.index, rolling_sharpe, color='#3b82f6', lw=1.5)
        ax.axhline(0, color='black', ls='--')
        ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe, where=(rolling_sharpe>0), color='#22c55e', alpha=0.3)
        ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe, where=(rolling_sharpe<0), color='#ef4444', alpha=0.3)
        
        ax.set_title(f"Rolling 6-Month Sharpe Ratio (XGB_Short Net of 10 bps Cost - {horizon} Horizon)", fontweight='bold', pad=12)
        ax.set_ylabel("Sharpe Ratio")
        ax.set_xlabel("Date")
        
        fig.tight_layout()
        plot_path = self.figures_dir / f"rolling_sharpe_{horizon}_feat{self.feature_size}.png"
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved rolling Sharpe plot to {plot_path}")
        
        if self.feature_size == '11':
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(rolling_sharpe.index, rolling_sharpe, color='#3b82f6', lw=1.5)
            ax.axhline(0, color='black', ls='--')
            ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe, where=(rolling_sharpe>0), color='#22c55e', alpha=0.3)
            ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe, where=(rolling_sharpe<0), color='#ef4444', alpha=0.3)
            ax.set_title(f"Rolling 6-Month Sharpe Ratio (XGB_Short Net of 10 bps Cost - {horizon} Horizon)", fontweight='bold', pad=12)
            ax.set_ylabel("Sharpe Ratio")
            ax.set_xlabel("Date")
            plt.tight_layout()
            plt.savefig(self.figures_dir / f"rolling_sharpe_{horizon}.png", dpi=300)
            plt.close()
            
            if horizon == '1d':
                fig_path = self.figures_dir / "rolling_sharpe.png"
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(rolling_sharpe.index, rolling_sharpe, color='#3b82f6', lw=1.5)
                ax.axhline(0, color='black', ls='--')
                ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe, where=(rolling_sharpe>0), color='#22c55e', alpha=0.3)
                ax.fill_between(rolling_sharpe.index, 0, rolling_sharpe, where=(rolling_sharpe<0), color='#ef4444', alpha=0.3)
                ax.set_title("Rolling 6-Month Sharpe Ratio (XGB_Short Net of 10 bps Cost)", fontweight='bold', pad=12)
                ax.set_ylabel("Sharpe Ratio")
                ax.set_xlabel("Date")
                plt.tight_layout()
                plt.savefig(fig_path, dpi=300)
                plt.close()
