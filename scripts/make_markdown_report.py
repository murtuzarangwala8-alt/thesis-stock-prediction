import json

def main():
    json_path = "data/processed/feature_health_report.json"
    md_path = "C:/Users/murta/.gemini/antigravity/brain/1782696f-ee56-4d5f-b6b7-55a09dbe2558/feature_health_report.md"
    
    with open(json_path, "r") as f:
        data = json.load(f)
        
    md = []
    md.append("# Feature Health & Quality Report")
    md.append("")
    md.append("This report presents key quality, distribution, and predictive stats for all selected features across the 4 modalities.")
    md.append("")
    
    # Summary stats
    total_feats = len(data)
    healthy_feats = sum(1 for x in data if x["status"] == "HEALTHY")
    warning_feats = sum(1 for x in data if x["status"] == "WARNING")
    
    md.append("## Summary Statistics")
    md.append(f"- **Total Features Checked**: {total_feats}")
    md.append(f"- **Healthy Features (No Missing/Infs)**: {healthy_feats}")
    md.append(f"- **Features with Warnings**: {warning_feats}")
    md.append("")
    
    # Table header
    md.append("## Feature Details")
    md.append("")
    md.append("| Feature | Modality | Missing % | Infs | Mean | Std | Min | Median | Max | Skew | Kurt | Pearson Corr | Spearman Corr | Status |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for x in data:
        md.append(
            f"| `{x['feature']}` | {x['modality']} | {x['missing_pct']:.2f}% | {x['inf_count']} | "
            f"{x['mean']:.4f} | {x['std']:.4f} | {x['min']:.4f} | {x['median']:.4f} | {x['max']:.4f} | "
            f"{x['skewness']:.2f} | {x['kurtosis']:.2f} | {x['pearson_corr']:.4f} | {x['spearman_corr']:.4f} | "
            f"**{x['status']}** |"
        )
        
    with open(md_path, "w") as f:
        f.write("\n".join(md))
        
    print(f"Markdown report written to: {md_path}")

if __name__ == "__main__":
    main()
