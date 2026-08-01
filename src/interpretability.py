import pandas as pd
import numpy as np
import pickle
import shap
from pathlib import Path
from .utils import setup_logger
import matplotlib.pyplot as plt

logger = setup_logger("Interpretability")

class InterpretabilityEngine:
    """
    Generates SHAP (Shapley Additive exPlanations) values to extract 
    non-linear feature interactions and true causal feature importance.
    """
    def __init__(self, data_dir: Path, results_dir: Path, figures_dir: Path, feature_size='11'):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.figures_dir = figures_dir
        self.models_dir = self.results_dir / "models"
        self.feature_size = feature_size
        
        self.features_11 = [
            'mom_21d_rank', 'mom_252d_rank', 'vol_21d_rank', 'rsi_14_rank', 
            'beta_mkt_5f_252d_rank', 'beta_smb_5f_252d_rank', 'beta_hml_5f_252d_rank',
            'beta_rmw_5f_252d_rank', 'beta_cma_5f_252d_rank',
            'book_to_market_rank', 'quality_score_rank'
        ]
        
        self.features_50 = [
            'mom_1d_rank', 'mom_5d_rank', 'mom_21d_rank', 'mom_63d_rank', 'mom_126d_rank', 'mom_252d_rank',
            'rsi_14_rank', 'macd_hist_rank', 'bb_pct_b_rank',
            'vol_5d_rank', 'vol_21d_rank', 'vol_63d_rank', 'vol_126d_rank', 'vol_ratio_rank',
            'beta_252d_rank',
            'earnings_yield_rank', 'book_to_market_rank', 'quality_score_rank',
            'pe_ratio_rank', 'pb_ratio_rank', 'return_on_equity_rank', 'return_on_assets_rank',
            'return_on_inv_capital_rank', 'sales_growth_rank', 'net_profit_margin_rank',
            'oper_margin_rank', 'asset_turnover_rank', 'debt_to_equity_fundamentals_rank',
            'debt_to_assets_rank', 'piotroski_f_score_rank',
            'buy_ratio_rank', 'pt_upside_rank', 'best_analyst_rating_rank', 'price_target_rank', 'analyst_count_rank',
            'oil_wti_backup_rank', 'oil_brent_backup_rank', 'gold_backup_rank', 'silver_backup_rank',
            'dollar_index_backup_rank', 'eurusd_backup_rank', 'usdjpy_backup_rank',
            'vix_rank', 'vxn_backup_rank', 'vvix_backup_rank',
            'yield_3m_backup_rank', 'yield_5y_backup_rank', '10y_yield_rank', 'yield_30y_backup_rank',
            'fed_funds_rank'
        ]
        
        self.features_80 = self.features_50 + [
            'acct_rcv_rank', 'altman_z_score_rank', 'total_assets_rank', 'bs_tot_liab2_rank',
            'capital_expend_rank', 'cf_cash_from_oper_rank', 'cf_dvd_paid_rank', 'cf_free_cash_flow_rank',
            'cur_mkt_cap_rank', 'ebit_rank', 'ebitda_rank', 'eqy_float_rank', 'eqy_sh_out_rank',
            'ev_to_t12m_ebitda_rank', 'free_cash_flow_per_sh_rank', 'is_eps_rank', 'is_oper_inc_rank',
            'long_term_borrow_rank', 'net_income_rank', 'px_volume_rank', 'sales_rev_turn_rank',
            'short_term_borrow_rank', 'tax_rate_reported_rank', 'tot_common_eqy_rank',
            'tot_return_index_gross_dvds_rank', 'eps_trailing_rank', 'ebitda_to_revenue_rank',
            'px_to_sales_ratio_rank', 'px_to_free_cash_flow_rank', 'beta_mkt_5f_252d_rank'
        ]
        
        selected_path = self.data_dir / "processed" / "selected_features.json"
        
        if self.feature_size == '11':
            self.features = self.features_11
        elif self.feature_size == '50':
            self.features = self.features_50
        elif self.feature_size == '80':
            self.features = self.features_80
        elif self.feature_size in ['technical', 'fundamental', 'macro', 'sentiment', 'all_selected', 'selected', 'tech_fund', 'tech_fund_macro']:
            if selected_path.exists():
                logger.info(f"Loading feature size '{self.feature_size}' from {selected_path}")
                import json
                with open(selected_path, "r") as f:
                    sel_data = json.load(f)
                if self.feature_size == 'technical':
                    self.features = sel_data["tech_cols"]
                elif self.feature_size == 'tech_fund':
                    self.features = sel_data["tech_cols"] + sel_data["fund_cols"]
                elif self.feature_size == 'tech_fund_macro':
                    self.features = sel_data["tech_cols"] + sel_data["fund_cols"] + sel_data["macro_cols"]
                elif self.feature_size == 'fundamental':
                    self.features = sel_data["fund_cols"]
                elif self.feature_size == 'macro':
                    self.features = sel_data["macro_cols"]
                elif self.feature_size == 'sentiment':
                    self.features = sel_data["sent_cols"]
                else: # 'all_selected' or 'selected'
                    self.features = sel_data["tech_cols"] + sel_data["fund_cols"] + sel_data["macro_cols"] + sel_data["sent_cols"]
            else:
                raise FileNotFoundError(f"Feature selection file not found at {selected_path}. Run select_features.py first.")
        else:
            raise ValueError(f"Invalid feature_size: {feature_size}. Must be '11', '50', '80', 'technical', 'tech_fund', 'tech_fund_macro', 'fundamental', 'macro', 'sentiment', or 'all_selected'.")

    def run_shap_analysis(self, horizon='1d'):
        """Runs SHAP analysis on the trained XGBoost model using the OOS test set."""
        logger.info(f"Loading Out-Of-Sample data and XGBoost model for SHAP analysis (Horizon: {horizon})...")
        
        oos_path = self.data_dir / "processed" / f"oos_predictions_{horizon}_feat{self.feature_size}.parquet"
        if not oos_path.exists():
            oos_path = self.data_dir / "processed" / f"oos_predictions_{horizon}.parquet"
            if not oos_path.exists():
                logger.error(f"OOS predictions for horizon {horizon} and feature size {self.feature_size} not found. Run MLModels first.")
                return
            
        test_df = pd.read_parquet(oos_path)
        X_test = test_df[self.features]
        
        xgb_path = self.models_dir / f"xgb_model_{horizon}_feat{self.feature_size}.pkl"
        if not xgb_path.exists():
            xgb_path = self.models_dir / f"xgb_model_{horizon}.pkl"
            if not xgb_path.exists():
                xgb_path = self.models_dir / "xgb_model.pkl"
                if not xgb_path.exists():
                    logger.error(f"XGBoost model not found for horizon {horizon} and feature size {self.feature_size}.")
                    return
            
        with open(xgb_path, "rb") as f:
            xgb_model = pickle.load(f)
            
        logger.info("Calculating SHAP values (sampled to 10,000 for tractability)...")
        # Sample but ensure we don't exceed actual data length if smaller than 10000
        n_samples = min(len(X_test), 10000)
        X_sample = X_test.sample(n=n_samples, random_state=42)
        
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_sample)
        
        # Check if SHAP returns list for binary classification, extract probability dimension
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
            
        # SHAP Summary Plot (Beeswarm)
        logger.info(f"Generating SHAP Summary Plot for horizon {horizon}...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False, plot_type="dot")
        plt.title(f"SHAP Summary Plot (XGBoost - {horizon} Horizon)\nGlobal Feature Importance & Non-Linear Effects", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.figures_dir / f"shap_summary_plot_{horizon}_feat{self.feature_size}.png", dpi=300, bbox_inches='tight')
        if self.feature_size == '11':
            plt.savefig(self.figures_dir / f"shap_summary_plot_{horizon}.png", dpi=300, bbox_inches='tight')
            if horizon == '1d':
                plt.savefig(self.figures_dir / "shap_summary_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # SHAP Bar Plot (Global Mean Absolute SHAP)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False, plot_type="bar")
        plt.title(f"SHAP Global Feature Importance (Mean Absolute SHAP - {horizon} Horizon)", fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.figures_dir / f"shap_bar_plot_{horizon}_feat{self.feature_size}.png", dpi=300, bbox_inches='tight')
        if self.feature_size == '11':
            plt.savefig(self.figures_dir / f"shap_bar_plot_{horizon}.png", dpi=300, bbox_inches='tight')
            if horizon == '1d':
                plt.savefig(self.figures_dir / "shap_bar_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved SHAP plots for horizon {horizon}.")
