import pandas as pd
import numpy as np
import json
import os
from sklearn.ensemble import RandomForestRegressor

# FEATURE CLASSIFICATION FIX (Audit Fix M5)
# ============================================
# Corrected misclassified keywords:
# - 'beta', 'alpha' moved from tech → fund (Fama-French factor loadings)
# - Added explicit fundamental keywords to prevent fund features landing in tech
# - 'rf' moved to _IGNORE_COLS (it's a label, not a feature)

_TECH_KEYWORDS = [
    "mom", "momentum", "rsi", "macd", "roc",
    "atr", "bb_", "ema", "sma", "ret_", "return_",
    "price_", "close_", "range_", "stoch", "obv", "adx", "cci",
    "willr", "mfi", "ultosc", "trix", "dmi", "aroon", "kama",
    "vol_", "volatility_",  # Explicit underscore to avoid matching 'volume'
]
_FUND_KEYWORDS = [
    "beta", "alpha", "pe_", "pb_", "roe", "roa", "eps",
    "margin", "debt", "dividend", "book", "earn", "revenue",
    "asset", "equity", "cash_flow", "leverage", "turnover_ratio",
    "market_cap", "current_ratio", "quick_ratio",
]
_MACRO_KEYWORDS = [
    "vix", "yield", "rate", "fed", "treasury", "spread", "credit",
    "oil", "gold", "copper", "dollar", "dxy", "eur", "usd", "gbp",
    "cpi", "ppi", "gdp", "unemployment", "ism", "pmi", "macro",
    "term_", "risk_free",
]
_SENT_KEYWORDS = [
    "sentiment", "sent_", "news_", "social_", "mood", "opinion",
]
_IGNORE_COLS = {
    "date", "ticker", "symbol", "permno", "gvkey",
    "target_ret_1d", "target_ret_21d", "target_ret_126d",
    "ret", "ret_1d", "ret_21d", "close", "open", "high", "low",
    "volume", "adj_close", "adj_close_1d", "year", "month", "day",
    "rf",  # Risk-free rate — label, not a feature
}

def main():
    data_path = "data/processed/master_panel_features.parquet"
    out_path = "data/processed/selected_features.json"
    
    print(f"Reading dataset from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Dataset shape: {df.shape}")
    
    # 1. Classify candidate columns (drop if missing > 10%)
    print("Checking missing values on all candidate features...")
    missing_pcts = df.isna().mean()
    all_feat = [
        c for c in df.columns
        if c.lower() not in _IGNORE_COLS
        and not any(k in c.lower() for k in ("target", "forward", "fwd", "future"))
        and missing_pcts[c] <= 0.10
    ]
    dropped_missing = [
        c for c in df.columns
        if c.lower() not in _IGNORE_COLS
        and not any(k in c.lower() for k in ("target", "forward", "fwd", "future"))
        and missing_pcts[c] > 0.10
    ]
    print(f"Dropped {len(dropped_missing)} features due to missing values > 10% (e.g. {dropped_missing[:5]})")
    
    def _match(col: str, keywords: list) -> bool:
        cl = col.lower()
        return any(k in cl for k in keywords)
        
    sent_candidates = [c for c in all_feat if _match(c, _SENT_KEYWORDS)]
    macro_candidates = [c for c in all_feat if c not in sent_candidates and _match(c, _MACRO_KEYWORDS)]
    tech_candidates = [c for c in all_feat if c not in sent_candidates and c not in macro_candidates and _match(c, _TECH_KEYWORDS)]
    # Explicit fundamental keyword match + catch-all for unmatched features
    fund_explicit = [c for c in all_feat if c not in sent_candidates and c not in macro_candidates and c not in tech_candidates and _match(c, _FUND_KEYWORDS)]
    fund_catchall = [c for c in all_feat if c not in sent_candidates and c not in macro_candidates and c not in tech_candidates and c not in fund_explicit]
    fund_candidates = fund_explicit + fund_catchall
    
    print("\nInitial candidate counts:")
    print(f"  Technical Candidates  : {len(tech_candidates)}")
    print(f"  Fundamental Candidates: {len(fund_candidates)}")
    print(f"  Macro Candidates      : {len(macro_candidates)}")
    print(f"  Sentiment Candidates  : {len(sent_candidates)}")

    groups = {
        "technical": tech_candidates,
        "fundamental": fund_candidates,
        "macro": macro_candidates,
        "sentiment": sent_candidates
    }
    
    # Fill NA for correlation calculations
    df_clean = df.sample(n=min(300000, len(df)), random_state=42).copy()
    for col in all_feat:
        df_clean[col] = df_clean[col].fillna(0)
    df_clean["target_ret_1d"] = df_clean["target_ret_1d"].fillna(0)
    
    # 2. Filter 1: Rank Constraint (Deduplicate Raw vs Rank)
    # If both X and X_rank exist in the group, drop X
    for gname, cols in groups.items():
        if gname == "sentiment":
            continue
        to_remove = []
        for col in cols:
            if col.endswith("_rank"):
                raw_name = col[:-5]
                if raw_name in cols:
                    to_remove.append(raw_name)
        groups[gname] = [c for c in cols if c not in to_remove]
        print(f"  After raw/rank deduplication in {gname}: {len(groups[gname])} features remaining (removed {len(to_remove)} raw counterparts)")

    # 3. Filter 2: Collinearity Pruning
    # Within each modality, compute correlations. If |corr| > 0.85, drop the one with lower target corr.
    for gname, cols in groups.items():
        if gname == "sentiment" or len(cols) <= 1:
            continue
        print(f"  Applying collinearity filter on {gname}...")
        # Compute correlation matrix
        corr_mat = df_clean[cols].corr().abs()
        # Compute correlation with target
        target_corr = df_clean[cols].corrwith(df_clean["target_ret_1d"]).abs()
        
        dropped = set()
        for i in range(len(cols)):
            col_i = cols[i]
            if col_i in dropped:
                continue
            for j in range(i + 1, len(cols)):
                col_j = cols[j]
                if col_j in dropped:
                    continue
                if corr_mat.at[col_i, col_j] > 0.85:
                    # Drop the one with lower target correlation
                    if target_corr[col_i] >= target_corr[col_j]:
                        dropped.add(col_j)
                    else:
                        dropped.add(col_i)
                        break
        groups[gname] = [c for c in cols if c not in dropped]
        print(f"  After collinearity pruning in {gname}: {len(groups[gname])} features remaining (removed {len(dropped)} collinear features)")

    # 4. Filter 3: Predictive Scoring via RandomForest
    selected = {}
    target_k = {
        "technical": 20,
        "fundamental": 30,
        "macro": 15,
        "sentiment": 2
    }
    
    print("\nTraining feature selection model (Random Forest Regressor)...")
    # FEATURE SELECTION LEAKAGE FIX (Audit Fix C3)
    # ==============================================
    # Restrict RF importance scoring to in-sample training period ONLY.
    # Previously drew from the entire dataset (2015-2025), leaking OOS
    # information into feature selection and invalidating walk-forward tests.
    if 'date' in df_clean.columns:
        cutoff = pd.Timestamp('2020-01-01')
        in_sample = df_clean[pd.to_datetime(df_clean['date']) < cutoff]
    elif 'year' in df_clean.columns:
        in_sample = df_clean[df_clean['year'] < 2020]
    else:
        # Fallback: use first 60% of data as a conservative in-sample estimate
        in_sample = df_clean.iloc[:int(len(df_clean) * 0.6)]
    print(f"  Using in-sample data only for RF importance: {len(in_sample):,} rows (pre-2020)")
    train_sample = in_sample.sample(n=min(50000, len(in_sample)), random_state=42)
    y = train_sample["target_ret_1d"].values
    
    for gname, cols in groups.items():
        k = target_k[gname]
        if len(cols) <= k:
            selected[gname] = cols
            print(f"  Selected all {len(cols)} features for {gname} (requested {k})")
            continue
            
        X = train_sample[cols].values
        rf = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        importances = rf.feature_importances_
        
        # Rank by importance
        ranked_indices = np.argsort(importances)[::-1]
        selected_cols = [cols[idx] for idx in ranked_indices[:k]]
        selected[gname] = selected_cols
        print(f"  Selected top {k} features for {gname} based on RF Importance")

    # Output selected columns to JSON
    output_data = {
        "tech_cols": selected["technical"],
        "fund_cols": selected["fundamental"],
        "macro_cols": selected["macro"],
        "sent_cols": selected["sentiment"]
    }
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
        
    print(f"\nSuccessfully wrote selected features JSON to: {out_path}")
    print("Summary of selected features:")
    for k, v in output_data.items():
        print(f"  {k}: {len(v)} features")

if __name__ == "__main__":
    main()
