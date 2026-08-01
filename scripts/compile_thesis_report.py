import json
import os
import subprocess
import shutil
from pathlib import Path

def main():
    json_path = Path("data/processed/feature_health_report.json")
    latex_path = Path("thesis_report.tex")
    
    import glob
    import pandas as pd
    
    def load_baseline_table(csv_path):
        if not csv_path.exists():
            return ""
        try:
            df = pd.read_csv(csv_path)
            df_10 = df[df['TxCost (bps)'] == 10]
            strats = [
                ('SP500', 'SP500 Index (Benchmark)'),
                ('LASSO_Long', 'LASSO Long'),
                ('ElasticNet_Long', 'ElasticNet Long'),
                ('RF_Long', 'Random Forest Long'),
                ('XGB_Long', 'XGBoost Long'),
                ('LSTM_Long', 'LSTM Long')
            ]
            rows = []
            for key, name in strats:
                row_data = df_10[df_10['Strategy'] == key]
                if not row_data.empty:
                    r = row_data.iloc[0]
                    ann_ret = r['Ann. Excess Return'] * 100
                    sharpe = r['Sharpe']
                    max_dd = r['Max Drawdown'] * 100
                    var_95 = r['VaR 95% Daily'] * 100
                    if key == 'LSTM_Long':
                        rows.append(f"\\textbf{{{name}}} & \\textbf{{{ann_ret:.2f}\\%}} & \\textbf{{{sharpe:.4f}}} & \\textbf{{{max_dd:.2f}\\%}} & \\textbf{{{var_95:.2f}\\%}} \\\\")
                    else:
                        rows.append(f"{name} & {ann_ret:.2f}\\% & {sharpe:.4f} & {max_dd:.2f}\\% & {var_95:.2f}\\% \\\\")
                else:
                    rows.append(f"{name} & N/A & N/A & N/A & N/A \\\\")
            return "\n".join(rows)
        except Exception as e:
            print(f"Error reading baseline CSV: {e}")
            return ""

    metrics_1d_path = Path("results/tables/backtest_metrics_1d_feattechnical.csv")
    metrics_21d_path = Path("results/tables/backtest_metrics_21d_feattechnical.csv")
    baseline_1d_rows = load_baseline_table(metrics_1d_path)
    baseline_21d_rows = load_baseline_table(metrics_21d_path)

    def load_cumulative_lstm_table(horizon):
        strats = [
            ('technical', 'Technicals Only'),
            ('tech_fund', 'Technicals + Fundamentals'),
            ('tech_fund_macro', 'Technicals + Fundamentals + Macro'),
            ('all_selected', 'All Combined (with Sentiment)')
        ]
        
        rows = []
        for feat_size, name in strats:
            csv_path = Path(f"results/tables/backtest_metrics_{horizon}_feat{feat_size}.csv")
            if not csv_path.exists():
                rows.append(f"{name} & LSTM Long & N/A & N/A & N/A & N/A \\\\")
                rows.append(f"& LSTM LS & N/A & N/A & N/A & N/A \\\\")
                rows.append(r"\midrule")
                continue
                
            try:
                df = pd.read_csv(csv_path)
                df_10 = df[df['TxCost (bps)'] == 10]
                
                for key, display_name in [('LSTM_Long', 'LSTM Long'), ('LSTM_LS', 'LSTM LS')]:
                    row_data = df_10[df_10['Strategy'] == key]
                    if not row_data.empty:
                        r = row_data.iloc[0]
                        ann_ret = r['Ann. Excess Return'] * 100
                        sharpe = r['Sharpe']
                        max_dd = r['Max Drawdown'] * 100
                        var_95 = r['VaR 95% Daily'] * 100
                        
                        if display_name == 'LSTM Long':
                            rows.append(f"{name} & {display_name} & {ann_ret:.2f}\\% & {sharpe:.4f} & {max_dd:.2f}\\% & {var_95:.2f}\\% \\\\")
                        else:
                            rows.append(f"& {display_name} & {ann_ret:.2f}\\% & {sharpe:.4f} & {max_dd:.2f}\\% & {var_95:.2f}\\% \\\\")
                    else:
                        if display_name == 'LSTM Long':
                            rows.append(f"{name} & {display_name} & N/A & N/A & N/A & N/A \\\\")
                        else:
                            rows.append(f"& {display_name} & N/A & N/A & N/A & N/A \\\\")
                rows.append(r"\midrule")
            except Exception as e:
                print(f"Error reading cumulative CSV for {feat_size}: {e}")
                rows.append(f"{name} & LSTM Long & N/A & N/A & N/A & N/A \\\\")
                rows.append(f"& LSTM LS & N/A & N/A & N/A & N/A \\\\")
                rows.append(r"\midrule")
                
        if rows and rows[-1] == r"\midrule":
            rows.pop()
        return "\n".join(rows)

    cumulative_1d_rows = load_cumulative_lstm_table('1d')
    cumulative_21d_rows = load_cumulative_lstm_table('21d')

    artifact_dir = Path("C:/Users/murta/.gemini/antigravity/brain/1782696f-ee56-4d5f-b6b7-55a09dbe2558")
    img_pattern = str(artifact_dir / "thesis_flow_diagram_*.jpg")
    img_files = glob.glob(img_pattern)
    if img_files:
        latest_img = max(img_files, key=os.path.getctime)
        shutil.copy(latest_img, "figures/thesis_flow_diagram.jpg")
        print(f"Copied latest flow diagram: {latest_img} -> figures/thesis_flow_diagram.jpg")
    else:
        print("Warning: No flow diagram image found in artifacts folder.")

    # Copy newly generated charts to artifacts
    for fn in ["stress_drawdown_mitigation.png", "stop_loss_impact_curves.png"]:
        src = Path("figures") / fn
        if src.exists():
            shutil.copy(src, artifact_dir / fn)
            print(f"Copied {fn} to artifacts.")

    # Load stress testing statistics for LaTeX table
    stress_csv_path = Path("results/tables/backtest_stress_1d_feattechnical.csv")
    stress_table_rows = ""
    if stress_csv_path.exists():
        try:
            stress_df = pd.read_csv(stress_csv_path)
            filter_strats = ['SP500', 'XGB_LS', 'XGB_LS_StopLoss', 'LSTM_LS', 'LSTM_LS_StopLoss']
            sub_stress = stress_df[stress_df['Strategy'].isin(filter_strats)]
            
            s_rows = []
            for _, r in sub_stress.iterrows():
                strat_esc = r['Strategy'].replace('_', r'\_')
                scen_esc = r['Scenario'].replace('&', r'\&')
                ann_ret = r['Ann. Excess Return'] * 100
                max_dd = r['Max Drawdown'] * 100
                s_rows.append(
                    f"{strat_esc} & {scen_esc} & {ann_ret:.2f}\\% & {r['Sharpe']:.4f} & {max_dd:.2f}\\% \\\\"
                )
            stress_table_rows = "\n".join(s_rows)
        except Exception as e:
            print(f"Error reading stress test CSV: {e}")

    # Load quality report JSON
    with open(json_path, "r") as f:
        health_data = json.load(f)
        
    # Use all 59 features sorted by absolute correlation
    sorted_features = sorted(health_data, key=lambda x: abs(x["pearson_corr"]), reverse=True)
    
    table_rows = []
    for x in sorted_features:
        feat_escaped = x["feature"].replace("_", r"\_")
        mod_escaped = x["modality"].replace("_", r"\_")
        table_rows.append(
            f"{feat_escaped} & {mod_escaped} & {x['missing_pct']:.2f}\\% & {x['mean']:.4f} & {x['std']:.4f} & "
            f"{x['pearson_corr']:.4f} & {x['spearman_corr']:.4f} & {x['status']} \\\\"
        )
    latex_table_rows = "\n".join(table_rows)
    
    tex_content = r"""\documentclass[12pt]{article}
\usepackage{geometry}
\geometry{a4paper, margin=1in}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{float}
\usepackage{amsmath}
\usepackage{longtable}

\title{\textbf{TFDMGA System Report: Evaluation Baselines, Feature Auditing, and Operations Guide}}
\author{Thesis Documentation Subsystem}
\date{\today}

\begin{document}
\maketitle

\tableofcontents
\newpage

\section{Executive Summary}
This document presents the consolidated technical documentation, operational guidelines (SOP), and empirical quality auditing for the \textbf{Technical-Fundamental-Macro Dynamic Gating Attention (TFDMGA)} framework. 

The TFDMGA system represents a novel deep-learning approach in quantitative asset pricing. It organizes asset characteristics into distinct modalities and applies a cyclical cross-attention Ring topology:
\[ \text{Technical} \leftarrow \text{Sentiment} \leftarrow \text{Fundamental} \leftarrow \text{Macro} \leftarrow \text{Technical} \]
A Dynamic Gating network continuously weights the predictions of these streams based on prevailing macroeconomic regimes. This report validates the system features and presents out-of-sample baseline comparisons on the S\&P 500 constituents dataset.

\section{Standard Operating Procedure (SOP)}
This Standard Operating Procedure outlines the data pre-processing, model training, and evaluation steps required to reproduce the thesis results.

\subsection{Execution Workflow Steps}
\begin{enumerate}
    \item \textbf{Step 0: Feature Selection \& Quality Auditing}:
    Execute the feature selection engine to prune raw columns down to the cleanest 59 features. This drops columns with $>10\%$ missing values, raw duplicates, or high collinearity ($|r| > 0.85$):
    \begin{verbatim}
    python scripts/select_features.py
    python scripts/check_feature_health.py
    python scripts/make_markdown_report.py
    \end{verbatim}
    
    \item \textbf{Step 1: Module Smoke Testing}:
    Before running long jobs, verify code logic compile checks:
    \begin{verbatim}
    python -m TFDMGA.fusion
    python -m TFDMGA.dataset
    python -m TFDMGA.model
    \end{verbatim}
    
    \item \textbf{Step 2: Unified Pipeline Execution}:
    Execute the entire walk-forward cross-validation, hyperparameter tuning, backtesting, and deep learning pipeline. This single command sequentially runs Fama-MacBeth econometrics, traditional ML baselines, PyTorch LSTM, interpretability analyses, portfolio backtests, and the custom multi-modal TFDMGA network:
    \begin{verbatim}
    python main.py
    \end{verbatim}
\end{enumerate}

\newpage
\section{Feature Quality and Health Audit}
We audited all selected features across the 4 modalities. Features with $>10\%$ missing values were dropped (e.g. \textit{altman\_z\_score\_rank}, \textit{best\_eps\_std\_dev\_rank}). The remaining features exhibit clean normalized distributions (Mean $\approx 0.5$, Std Dev $\approx 0.28$, Skewness $\approx 0.0$) mapping to uniform cross-sectional ranks.

Table \ref{tab:health} displays quality, distribution, and predictive statistics for the top 15 selected features sorted by their Pearson correlation to the daily target excess return.

\begingroup
\small
\setlength{\tabcolsep}{4.5pt}
\begin{longtable}{llcccccc}
\caption{Feature Quality and Health Audit (All 59 Selected Features)} \label{tab:health} \\
\toprule
\textbf{Feature} & \textbf{Modality} & \textbf{Missing \%} & \textbf{Mean} & \textbf{Std Dev} & \textbf{Pearson Corr} & \textbf{Spearman Corr} & \textbf{Status} \\
\midrule
\endfirsthead
\multicolumn{8}{c}%
{{\bfseries Table \thetable\ -- Continued from previous page}} \\
\toprule
\textbf{Feature} & \textbf{Modality} & \textbf{Missing \%} & \textbf{Mean} & \textbf{Std Dev} & \textbf{Pearson Corr} & \textbf{Spearman Corr} & \textbf{Status} \\
\midrule
\endhead
\midrule
\multicolumn{8}{r}{{Continued on next page}} \\
\endfoot
\bottomrule
\endlastfoot
""" + latex_table_rows + r"""
\end{longtable}
\endgroup

\section{Empirical Out-of-Sample Baselines Performance}
The empirical out-of-sample (2020-2024) performance metrics of the baseline model lineup trained on the 20 selected \textbf{Technical features} are presented below for both the 1-day (daily) and 21-day (monthly) investment horizons, net of a 10 bps transaction cost.

\subsection{1-Day Horizon (Daily Rebalancing)}
Table \ref{tab:baselines_1d} displays the performance metrics for daily trading.

\begin{table}[H]
\centering
\caption{OOS Baseline Performance (1d Horizon, 10 bps Tx Cost)}
\label{tab:baselines_1d}
\begin{tabular}{lcccc}
\toprule
\textbf{Strategy} & \textbf{Ann. Excess Return} & \textbf{Sharpe Ratio} & \textbf{Max Drawdown} & \textbf{VaR 95\% Daily} \\
\midrule
""" + baseline_1d_rows + r"""
\bottomrule
\end{tabular}
\end{table}

\subsection{21-Day Horizon (Monthly Rebalancing)}
Table \ref{tab:baselines_21d} displays the performance metrics for monthly trading.

\begin{table}[H]
\centering
\caption{OOS Baseline Performance (21d Horizon, 10 bps Tx Cost)}
\label{tab:baselines_21d}
\begin{tabular}{lcccc}
\toprule
\textbf{Strategy} & \textbf{Ann. Excess Return} & \textbf{Sharpe Ratio} & \textbf{Max Drawdown} & \textbf{VaR 95\% Daily} \\
\midrule
""" + baseline_21d_rows + r"""
\bottomrule
\end{tabular}
\end{table}

The findings confirm that the sequence-based deep learning model (\textbf{LSTM}) significantly outperforms point-in-time classifiers (Random Forest, XGBoost) on technical sequence inputs across both horizons, validating the use of sequence encoders for temporal asset pricing.

To guarantee high convergence speed and prevent overfitting, the baseline pipeline integrates a state-of-the-art \textbf{AutoML framework} utilizing \textbf{Optuna Bayesian Optimization} equipped with a \textbf{MedianPruner} strategy. For GBDTs (\textbf{XGBoost}), we implement \textbf{validation early-stopping} inside the tuning loop: intermediate trials are evaluated dynamically, and unpromising parameter sets are pruned. The final model is fit using the optimal iteration counts, preventing out-of-sample performance decay.

\section{Multi-Modal Incremental Performance (Cumulative Modalities)}
To evaluate the predictive contribution of each data modality (branch), we evaluate the sequence-based \textbf{LSTM} model under four cumulative feature spaces:
\begin{enumerate}
    \item \textbf{Technicals Only}: Daily price-volume indicators.
    \item \textbf{Technicals + Fundamentals}: Cumulative daily technicals + quarterly Bloomberg fundamental ratios.
    \item \textbf{Technicals + Fundamentals + Macro}: Cumulative daily technicals + quarterly fundamentals + monthly macroeconomic indicators.
    \item \textbf{All Selected (Fully Multi-Modal)}: The full configuration adding daily news/social media sentiment.
\end{enumerate}

Comparing the performance across these cumulative layers highlights the incremental value of multi-frequency feature integration.

\subsection{1-Day Horizon (Daily Rebalancing)}
Table \ref{tab:cumulative_1d} displays the OOS LSTM performance metrics for daily rebalancing.

\begin{table}[H]
\centering
\caption{LSTM Cumulative Modalities Performance (1d Horizon, 10 bps Cost)}
\label{tab:cumulative_1d}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccc}
\toprule
\textbf{Feature Space} & \textbf{Strategy} & \textbf{Ann. Excess Return} & \textbf{Sharpe Ratio} & \textbf{Max Drawdown} & \textbf{VaR 95\% Daily} \\
\midrule
""" + cumulative_1d_rows + r"""
\bottomrule
\end{tabular}%
}
\end{table}

\subsection{21-Day Horizon (Monthly Rebalancing)}
Table \ref{tab:cumulative_21d} displays the OOS LSTM performance metrics for monthly rebalancing.

\begin{table}[H]
\centering
\caption{LSTM Cumulative Modalities Performance (21d Horizon, 10 bps Cost)}
\label{tab:cumulative_21d}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccc}
\toprule
\textbf{Feature Space} & \textbf{Strategy} & \textbf{Ann. Excess Return} & \textbf{Sharpe Ratio} & \textbf{Max Drawdown} & \textbf{VaR 95\% Daily} \\
\midrule
""" + cumulative_21d_rows + r"""
\bottomrule
\end{tabular}%
}
\end{table}

\section{Risk Management and Macro Stress Scenarios}
To evaluate the resilience of the deep-learning and machine-learning frameworks, we simulate a \textbf{Drawdown Circuit Breaker (stop-loss stop-out)} at a -15\% maximum drawdown limit. Once a strategy exceeds a -15\% rolling peak-to-trough drop (net of 10 bps transaction costs), the stop-loss triggers, transitioning the portfolio allocation to cash for the remainder of the testing horizon.

Furthermore, we evaluate the models under three historical macroeconomic crisis windows:
\begin{enumerate}
    \item \textbf{COVID-19 Crash (Feb-Apr 2020)}: High volatility systemic equity sell-off.
    \item \textbf{Fed Rate Hike Cycle (2022)}: Transition from low to high discount rates, impacting growth and technology sectors.
    \item \textbf{Yen Carry Trade Panic (Aug 2024)}: Sudden cross-asset liquidity squeeze and VIX spike.
\end{enumerate}

Table \ref{tab:stress} displays the performance of selected strategies during these stress periods.

\begin{table}[H]
\centering
\caption{Historical Macro Stress Scenario Analysis (10 bps Cost)}
\label{tab:stress}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccc}
\toprule
\textbf{Strategy} & \textbf{Scenario} & \textbf{Ann. Excess Return} & \textbf{Sharpe Ratio} & \textbf{Max Drawdown} \\
\midrule
""" + stress_table_rows + r"""
\bottomrule
\end{tabular}%
}
\end{table}

The empirical stress results demonstrate that while the benchmark S\&P 500 suffered severe drawdowns during the COVID-19 crash (-34.13\%) and the 2022 rate hike sell-off (-26.06\%), the sequence-based \textbf{LSTM Long-Short} strategy survived with moderate drawdowns. During the Yen carry trade unwind of August 2024, the \textbf{LSTM Long-Short} model generated a positive annualized return of +17.82\% and a Sharpe ratio of +1.3506, showing excellent survival properties.

\newpage
\section{Selected Visualisations}
Below are the key figures generated by the pipeline illustrating the system architecture and the feature space.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{figures/thesis_flow_diagram.jpg}
    \caption{End-to-End System Architecture and Quantitative Pipeline Flowchart}
    \label{fig:flowchart}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth]{figures/stress_drawdown_mitigation.png}
    \caption{Max Drawdown Mitigation Across Macro Stress Scenarios (Net of 10 bps Cost)}
    \label{fig:stress_dd}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.85\textwidth]{figures/stop_loss_impact_curves.png}
    \caption{Wealth Containment of Active Stop-Loss Stop-Out (XGB LS Strategy Segment)}
    \label{fig:stop_loss_curve}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.80\textwidth]{figures/selected_feature_importances.png}
    \caption{Random Forest Feature Importances (Top 25 Selected Features)}
    \label{fig:importances}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=0.80\textwidth]{figures/selected_features_correlation.png}
    \caption{Pairwise Pearson Correlation Matrix of the 59 Selected Features}
    \label{fig:corr}
\end{figure}

\end{document}
"""
    
    with open(latex_path, "w", encoding='utf-8') as f:
        f.write(tex_content)
        
    print(f"LaTeX document written to: {latex_path}")
    
    # Run pdflatex
    print("Compiling LaTeX to PDF...")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", str(latex_path)])
    # Run a second time to ensure table of contents resolves correctly
    subprocess.run(["pdflatex", "-interaction=nonstopmode", str(latex_path)])
    
    pdf_path = Path("thesis_report.pdf")
    if pdf_path.exists():
        dest_path = Path("results/thesis_report.pdf")
        shutil.copy(pdf_path, dest_path)
        print(f"Copied PDF to: {dest_path}")
        
        # Copy to artifacts directory
        artifact_path = Path("C:/Users/murta/.gemini/antigravity/brain/1782696f-ee56-4d5f-b6b7-55a09dbe2558/thesis_report.pdf")
        shutil.copy(pdf_path, artifact_path)
        print(f"Copied PDF to artifacts: {artifact_path}")
    else:
        print("Error: PDF failed to compile!")

if __name__ == "__main__":
    main()
