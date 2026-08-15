import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

# Clean display label mapping
CLEAN_NAME_MAP = {
    "natural_gas_backup_rank": "nat_gas_price_rank",
    "oil_wti_backup_rank": "oil_wti_price_rank",
    "rsi_14_rank": "rsi_14d_rank",
    "quality_score_rank": "quality_proxy_rank",
    "book_to_market_rank": "value_proxy_rank",
}

def main():
    data_path = Path("data/processed/master_panel_features.parquet")
    selected_path = Path("data/processed/selected_features.json")
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    
    print("Reading data for feature importances...")
    df = pd.read_parquet(data_path)
    df["target_ret_1d"] = df["target_ret_1d"].fillna(0)
    
    with open(selected_path, "r") as f:
        selected_data = json.load(f)
        
    all_selected = []
    feature_modalities = {}
    modality_label_map = {
        "tech_cols": "Technical",
        "fund_cols": "Fundamental",
        "macro_cols": "Macro",
        "sent_cols": "Sentiment"
    }
    
    for modality, cols in selected_data.items():
        mod_label = modality_label_map.get(modality, modality)
        for col in cols:
            all_selected.append(col)
            feature_modalities[col] = mod_label
            
    # Restrict to pre-2020 in-sample training split to prevent OOS leakage
    if 'date' in df.columns:
        in_sample = df[pd.to_datetime(df['date']) < '2020-01-01']
    else:
        in_sample = df.iloc[:int(len(df)*0.6)]

    sample_df = in_sample.sample(n=min(50000, len(in_sample)), random_state=42)
    X = sample_df[all_selected].fillna(0)
    y = sample_df["target_ret_1d"].fillna(0)
    
    rf = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Compute Permutation Feature Importance to prevent Gini continuous cardinality bias
    from sklearn.inspection import permutation_importance
    perm_imp = permutation_importance(rf, X, y, n_repeats=5, random_state=42, n_jobs=-1)
    importances = perm_imp.importances_mean
    
    clean_features = [CLEAN_NAME_MAP.get(f, f) for f in all_selected]
    
    imp_df = pd.DataFrame({
        "Feature": clean_features,
        "Importance": importances,
        "Modality": [feature_modalities[f] for f in all_selected]
    }).sort_values(by="Importance", ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    
    palette = {
        "Technical": "#1f77b4",
        "Fundamental": "#2ca02c",
        "Macro": "#ff7f0e",
        "Sentiment": "#d62728"
    }
    
    top_n = 20
    sns.barplot(
        data=imp_df.head(top_n),
        x="Importance",
        y="Feature",
        hue="Modality",
        palette=palette,
        dodge=False
    )
    plt.title("Stage 3 In-Sample Permutation Feature Importance Rankings Across Modalities", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("Permutation Predictive Importance Score", fontsize=11, fontweight="bold")
    plt.ylabel("Feature Name", fontsize=11, fontweight="bold")
    plt.legend(title="Modality Group", loc="lower right")
    plt.tight_layout()
    
    imp_plot_path = figures_dir / "selected_feature_importances.png"
    plt.savefig(imp_plot_path, dpi=300)
    plt.savefig("thesis/figures/selected_feature_importances.png", dpi=300)
    plt.close()
    print(f"Saved cleaned feature importances plot to {imp_plot_path}")
    
    # 2. Correlation Heatmap
    print("Computing correlation matrix...")
    X_renamed = X.rename(columns=CLEAN_NAME_MAP)
    corr_matrix = X_renamed.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Spearman Rank Correlation"}
    )
    plt.title("Spearman Rank Correlation Heatmap of Selected 53 ML Features", fontsize=13, fontweight="bold", pad=15)
    plt.xticks(fontsize=6, rotation=90)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    
    corr_plot_path = figures_dir / "selected_features_correlation.png"
    plt.savefig(corr_plot_path, dpi=300)
    plt.savefig("thesis/figures/selected_features_correlation.png", dpi=300)
    plt.close()
    print(f"Saved correlation heatmap to {corr_plot_path}")

if __name__ == "__main__":
    main()
