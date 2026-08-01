import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from .utils import setup_logger, export_csv_table, export_latex_table
import matplotlib.pyplot as plt

logger = setup_logger("BaselineModels")

class BaselineModels:
    """
    Executes baseline econometric regressions, specifically the 
    Fama-MacBeth (1973) two-step procedure with Newey-West standard errors.
    Tests the traditional Fama-French 3-Factor asset pricing model.
    """
    def __init__(self, data_dir: Path, results_dir: Path, figures_dir: Path):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.figures_dir = figures_dir
        self.tables_dir = self.results_dir / "tables"
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        self.features_path = self.data_dir / "processed" / "master_panel_features.parquet"

        # Fama-French 3-Factor betas (estimated in the first-pass rolling regressions)
        self.factor_betas = ['beta_mkt_252d', 'beta_smb_252d', 'beta_hml_252d']

    def run_fama_macbeth(self):
        """Runs the Fama-MacBeth 2-step regression for both 3-Factor and 5-Factor models."""
        logger.info("Loading master panel for Fama-MacBeth regressions...")
        df = pd.read_parquet(self.features_path)
        
        # Dependent variable: contemporaneous daily excess return
        df['excess_ret_contemp'] = df['ret'] - df['rf']
        
        # FAMA-MACBETH ENDOGENEITY FIX (Audit Fix C5)
        # ==============================================
        # Lag all factor betas by 1 day per ticker. Rolling 252d betas
        # computed through day t include day t's return, creating mechanical
        # endogeneity when used to explain day t's excess return.
        # Using t-1 betas ensures strict predictive (not contemporaneous)
        # conditioning in the 2nd-pass cross-sectional regression.
        all_beta_cols = [
            'beta_mkt_252d', 'beta_smb_252d', 'beta_hml_252d',
            'beta_mkt_5f_252d', 'beta_smb_5f_252d', 'beta_hml_5f_252d',
            'beta_rmw_5f_252d', 'beta_cma_5f_252d'
        ]
        existing_beta_cols = [c for c in all_beta_cols if c in df.columns]
        df = df.sort_values(['ticker', 'date'])
        for col in existing_beta_cols:
            df[col] = df.groupby('ticker')[col].shift(1)
        df = df.dropna(subset=existing_beta_cols)
        logger.info(f"Applied 1-day lag to {len(existing_beta_cols)} beta columns to prevent endogeneity.")
        
        unique_dates = sorted(df['date'].unique())
        
        # Define models
        models_config = {
            '3F': {
                'betas': ['beta_mkt_252d', 'beta_smb_252d', 'beta_hml_252d'],
                'suffix': '3f'
            },
            '5F': {
                'betas': ['beta_mkt_5f_252d', 'beta_smb_5f_252d', 'beta_hml_5f_252d', 'beta_rmw_5f_252d', 'beta_cma_5f_252d'],
                'suffix': '5f'
            }
        }
        
        for name, config in models_config.items():
            logger.info(f"\n--- Running Fama-MacBeth for Fama-French {name} Model ---")
            betas = config['betas']
            suffix = config['suffix']
            
            daily_coefficients = []
            dates_processed = []
            
            logger.info(f"Step 2 (Cross-Sectional): Running {len(unique_dates)} daily OLS regressions on {name} betas...")
            for d in unique_dates:
                day_data = df[df['date'] == d]
                day_data = day_data.dropna(subset=betas + ['excess_ret_contemp'])
                
                if len(day_data) < 50:  # Minimum cross-section size
                    continue
                    
                X_day = sm.add_constant(day_data[betas])
                y_day = day_data['excess_ret_contemp']
                
                try:
                    res = sm.OLS(y_day, X_day).fit()
                    daily_coefficients.append(res.params)
                    dates_processed.append(d)
                except Exception:
                    continue
                    
            coef_df = pd.DataFrame(daily_coefficients, index=dates_processed)
            
            logger.info("Step 3 (Time-Series): Averaging risk premiums and applying Newey-West correction...")
            T_fm = len(coef_df)
            optimal_nw_lags = int(np.floor(4 * (T_fm / 100) ** (2 / 9)))
            logger.info(f"Optimal Newey-West lags selected: L*={optimal_nw_lags}")
            
            fm_results = {}
            for col in coef_df.columns:
                const_model = sm.OLS(coef_df[col], np.ones(len(coef_df)))
                const_fit = const_model.fit(cov_type='HAC', cov_kwds={'maxlags': optimal_nw_lags})
                
                fm_results[col] = {
                    'Coefficient': const_fit.params.iloc[0],
                    'Std_Error': const_fit.bse.iloc[0],
                    't_Statistic': const_fit.tvalues.iloc[0],
                    'p_Value': const_fit.pvalues.iloc[0]
                }
                
            fm_summary = pd.DataFrame(fm_results).T
            logger.info(f"\n{name} Fama-MacBeth Results:")
            logger.info("\n" + fm_summary.to_string(float_format="%.5f"))
            
            # Calculate Economic Magnitude
            magnitude_df = pd.DataFrame({
                'FM_Coeff': fm_summary['Coefficient'],
                'Annualized_Spread_%': fm_summary['Coefficient'] * 252 * 100
            }).drop('const', errors='ignore')
            
            # Save Tables
            export_csv_table(fm_summary, self.tables_dir / f"fama_macbeth_{suffix}_results.csv")
            export_latex_table(fm_summary, self.tables_dir / f"fama_macbeth_{suffix}_results.tex")
            export_csv_table(magnitude_df, self.tables_dir / f"economic_magnitude_{suffix}.csv")
            export_latex_table(magnitude_df, self.tables_dir / f"economic_magnitude_{suffix}.tex")
            
            # Historical compatibility by copying 3f tables to baseline names
            if name == '3F':
                export_csv_table(fm_summary, self.tables_dir / "fama_macbeth_results.csv")
                export_latex_table(fm_summary, self.tables_dir / "fama_macbeth_results.tex")
                export_csv_table(magnitude_df, self.tables_dir / "economic_magnitude.csv")
                export_latex_table(magnitude_df, self.tables_dir / "economic_magnitude.tex")
                
            # Plot cumulative risk premiums for the model
            self._plot_cumulative_premiums(coef_df, name)

    def _plot_cumulative_premiums(self, coef_df: pd.DataFrame, model_name: str):
        """Plots the cumulative sum of factor premiums (risk premiums)."""
        factor_coefs = coef_df.drop(columns=["const"], errors="ignore")
        cum_premiums = factor_coefs.cumsum() * 100  # scale to %
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color palette and labels depending on model
        if model_name == '3F':
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            label_map = {
                'beta_mkt_252d': 'Market Factor (Mkt-RF)',
                'beta_smb_252d': 'Size Factor (SMB)',
                'beta_hml_252d': 'Value Factor (HML)'
            }
        else:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            label_map = {
                'beta_mkt_5f_252d': 'Market Factor (Mkt-RF)',
                'beta_smb_5f_252d': 'Size Factor (SMB)',
                'beta_hml_5f_252d': 'Value Factor (HML)',
                'beta_rmw_5f_252d': 'Profitability Factor (RMW)',
                'beta_cma_5f_252d': 'Investment Factor (CMA)'
            }
            
        for i, col in enumerate(cum_premiums.columns):
            ax.plot(cum_premiums.index, cum_premiums[col], label=label_map.get(col, col), lw=1.8, color=colors[i % len(colors)])
            
        ax.axhline(0, color='black', lw=1, linestyle='--')
        ax.set_ylabel("Cumulative Risk Premium (%)", fontweight='bold')
        ax.set_xlabel("Date", fontweight='bold')
        ax.set_title(f"Fama-MacBeth Cumulative Risk Premiums (FF {model_name})", fontweight='bold', pad=15)
        ax.legend(fontsize=9, loc='upper left', frameon=True)
        
        fig.tight_layout()
        suffix = model_name.lower()
        plot_path = self.figures_dir / f"fama_macbeth_cumulative_premiums_{suffix}.png"
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved figure to {plot_path}")
        
        # Historical compatibility for 3F
        if model_name == '3F':
            fig_path = self.figures_dir / "fama_macbeth_cumulative_premiums.png"
            plt.figure(figsize=(10, 6))
            for i, col in enumerate(factor_coefs.columns):
                plt.plot(cum_premiums.index, cum_premiums[col], label=label_map.get(col, col), lw=1.8, color=colors[i % len(colors)])
            plt.axhline(0, color='black', lw=1, linestyle='--')
            plt.ylabel("Cumulative Risk Premium (%)", fontweight='bold')
            plt.xlabel("Date", fontweight='bold')
            plt.title("Fama-MacBeth Cumulative Risk Premiums (FF 3-Factor)", fontweight='bold', pad=15)
            plt.legend(fontsize=9, loc='upper left', frameon=True)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=300)
            plt.close()
