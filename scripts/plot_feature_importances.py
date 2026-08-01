import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

def main():
    data_path = Path("data/processed/master_panel_features.parquet")
    selected_path = Path("data/processed/selected_features.json")
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    
    print("Reading data...")
    df = pd.read_parquet(data_path)
    df["target_ret_1d"] = df["target_ret_1d"].fillna(0)
    
    print("Loading selected features...")
    with open(selected_path, "r") as f:
        selected_data = json.load(f)
        
    # Plot 1: Feature Importances
    print("Computing feature importances...")
    all_selected = []
    feature_modalities = {}
    for modality, cols in selected_data.items():
        for col in cols:
            all_selected.append(col)
            feature_modalities[col] = modality
            
    sample_df = df.sample(n=min(100000, len(df)), random_state=42)
    X = sample_df[all_selected]
    y = sample_df["target_ret_1d"]
    
    rf = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    imp_df = pd.DataFrame({
        "Feature": all_selected,
        "Importance": importances,
        "Modality": [feature_modalities[f] for f in all_selected]
    }).sort_values(by="Importance", ascending=False)
    
    plt.figure(figsize=(12, 10))
    sns.set_theme(style="whitegrid")
    
    # Custom palette for modalities
    palette = {
        "tech_cols": "#1f77b4",
        "fund_cols": "#2ca02c",
        "macro_cols": "#ff7f0e",
        "sent_cols": "#d62728"
    }
    
    # Plot top 25 features for readability
    top_n = 25
    sns.barplot(
        data=imp_df.head(top_n),
        x="Importance",
        y="Feature",
        hue="Modality",
        palette=palette,
        dodge=False
    )
    plt.title(f"Top {top_n} Selected Feature Importances (Random Forest Regressor)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predictive Importance Score", fontsize=12, fontweight="bold")
    plt.ylabel("Feature Name", fontsize=12, fontweight="bold")
    plt.legend(title="Modality Group", loc="lower right")
    plt.tight_layout()
    
    imp_plot_path = figures_dir / "selected_feature_importances.png"
    plt.savefig(imp_plot_path, dpi=300)
    plt.close()
    print(f"Saved feature importances plot to {imp_plot_path}")
    
    # Plot 2: Correlation Heatmap
    print("Computing correlation matrix...")
    corr_matrix = X.corr()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Pearson Correlation Coefficient"}
    )
    plt.title("Correlation Matrix of Selected Multi-Modal Features", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(fontsize=6, rotation=90)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    
    corr_plot_path = figures_dir / "selected_features_correlation.png"
    plt.savefig(corr_plot_path, dpi=300)
    plt.close()
    print(f"Saved correlation heatmap to {corr_plot_path}")

if __name__ == "__main__":
    main()
