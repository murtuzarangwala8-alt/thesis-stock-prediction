import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Dynamic directory resolution (works in workspace and full thesis)
BASE_DIR = Path(__file__).resolve().parent
if BASE_DIR.name == 'code':
    BASE_DIR = BASE_DIR.parent

MODELS_DIR = BASE_DIR / "results" / "models"
DATA_DIR = BASE_DIR / "data"

def main():
    parser = argparse.ArgumentParser(description="Inspect model coefficients and constituents.")
    parser.add_argument("--horizon", type=str, default="1d", choices=["1d", "21d"], help="Target horizon.")
    parser.add_argument("--feature-size", type=str, default="11", choices=["11", "50", "80"], help="Feature space size.")
    args = parser.parse_args()

    horizon = args.horizon
    feature_size = args.feature_size
    suffix = f"_{horizon}_feat{feature_size}"
    
    print(f"==================================================")
    feat_display = "11 Technical Baseline" if feature_size == '11' else f"{feature_size}-Feature Space"
    print(f"  MODEL DIAGNOSTICS: Horizon={horizon} | Features={feat_display}")
    print(f"==================================================")

    features_11 = [
        'mom_21d_rank', 'mom_252d_rank', 'vol_21d_rank', 'rsi_14_rank', 
        'beta_mkt_5f_252d_rank', 'beta_smb_5f_252d_rank', 'beta_hml_5f_252d_rank',
        'beta_rmw_5f_252d_rank', 'beta_cma_5f_252d_rank',
        'book_to_market_rank', 'quality_score_rank'
    ]
    
    features_50 = [
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
    
    features_80 = features_50 + [
        'acct_rcv_rank', 'altman_z_score_rank', 'total_assets_rank', 'bs_tot_liab2_rank',
        'capital_expend_rank', 'cf_cash_from_oper_rank', 'cf_dvd_paid_rank', 'cf_free_cash_flow_rank',
        'cur_mkt_cap_rank', 'ebit_rank', 'ebitda_rank', 'eqy_float_rank', 'eqy_sh_out_rank',
        'ev_to_t12m_ebitda_rank', 'free_cash_flow_per_sh_rank', 'is_eps_rank', 'is_oper_inc_rank',
        'long_term_borrow_rank', 'net_income_rank', 'px_volume_rank', 'sales_rev_turn_rank',
        'short_term_borrow_rank', 'tax_rate_reported_rank', 'tot_common_eqy_rank',
        'tot_return_index_gross_dvds_rank', 'eps_trailing_rank', 'ebitda_to_revenue_rank',
        'px_to_sales_ratio_rank', 'px_to_free_cash_flow_rank', 'beta_mkt_5f_252d_rank'
    ]

    if feature_size == '11':
        features = features_11
    elif feature_size == '50':
        features = features_50
    else:
        features = features_80

    # 1. Inspect LASSO
    lasso_path = MODELS_DIR / f"lasso_model{suffix}.pkl"
    if not lasso_path.exists() and feature_size == '11' and horizon == '1d':
        lasso_path = MODELS_DIR / "lasso_model.pkl"
        
    print("\n=== LASSO MODEL COEFFICIENTS (NON-ZERO) ===")
    if lasso_path.exists():
        with open(lasso_path, "rb") as f:
            lasso = pickle.load(f)
        coefs = lasso.coef_[0]
        intercept = lasso.intercept_[0]
        print(f"Intercept: {intercept:.6f}")
        non_zero_coefs = [(feat, coef) for feat, coef in zip(features, coefs) if abs(coef) > 1e-5]
        non_zero_coefs = sorted(non_zero_coefs, key=lambda x: abs(x[1]), reverse=True)
        for feat, coef in non_zero_coefs[:25]:
            print(f"  {feat:<40}: {coef:.5f}")
        if len(non_zero_coefs) > 25:
            print(f"  ... and {len(non_zero_coefs) - 25} other non-zero features.")
    else:
        print(f"LASSO model file not found at {lasso_path}")

    # 2. Inspect XGBoost
    xgb_path = MODELS_DIR / f"xgb_model{suffix}.pkl"
    if not xgb_path.exists() and feature_size == '11' and horizon == '1d':
        xgb_path = MODELS_DIR / "xgb_model.pkl"

    print("\n=== XGBOOST FEATURE IMPORTANCES (TOP 15) ===")
    if xgb_path.exists():
        with open(xgb_path, "rb") as f:
            xgb_model = pickle.load(f)
        importances = xgb_model.feature_importances_
        sorted_imps = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        for feat, imp in sorted_imps[:15]:
            print(f"  {feat:<40}: {imp:.5f}")
    else:
        print(f"XGBoost model file not found at {xgb_path}")

    # 3. Analyze constituents
    oos_path = DATA_DIR / "processed" / f"oos_predictions_{horizon}_feat{feature_size}.parquet"
    if not oos_path.exists() and feature_size == '11':
        oos_path = DATA_DIR / "processed" / f"oos_predictions_{horizon}.parquet"
        if not oos_path.exists():
            oos_path = DATA_DIR / "processed" / "oos_predictions.parquet"

    print("\n=== PORTFOLIO CONSTITUENTS (OOS TEST PERIOD) ===")
    if oos_path.exists():
        df = pd.read_parquet(oos_path)
        print(f"Loaded out-of-sample predictions. Shape: {df.shape}")
        
        unique_dates = sorted(df['date'].unique())
        sample_dates = [unique_dates[0], unique_dates[len(unique_dates)//2], unique_dates[-1]]
        
        for d in sample_dates:
            d_str = pd.to_datetime(d).strftime('%Y-%m-%d')
            print(f"\nTrading Date: {d_str}")
            day_data = df[df['date'] == d]
            n_stocks = len(day_data)
            cutoff = int(np.ceil(n_stocks * 0.10))
            
            lasso_sorted = day_data.sort_values('pred_prob_lasso', ascending=False)
            longs = lasso_sorted.iloc[:cutoff]
            shorts = lasso_sorted.iloc[-cutoff:]
            
            print(f"  Top 5 Predicted LONG stocks (highest LASSO probability):")
            for idx, r in longs.head(5).iterrows():
                vol_val = r.get('px_volume', r.get('volume', 0.0))
                print(f"    {r['ticker']:<6} | Prob: {r['pred_prob_lasso']:.4f} | Daily Ret: {r['ret']*100:.2f}% | Book-to-Mkt Rank: {r.get('book_to_market_rank', np.nan):.4f} | Volume: {vol_val:,.0f}")
                
            print(f"  Top 5 Predicted SHORT stocks (lowest LASSO probability):")
            for idx, r in shorts.head(5).iterrows():
                vol_val = r.get('px_volume', r.get('volume', 0.0))
                print(f"    {r['ticker']:<6} | Prob: {r['pred_prob_lasso']:.4f} | Daily Ret: {r['ret']*100:.2f}% | Book-to-Mkt Rank: {r.get('book_to_market_rank', np.nan):.4f} | Volume: {vol_val:,.0f}")
    else:
        print(f"OOS predictions file not found at {oos_path}")

if __name__ == "__main__":
    main()
