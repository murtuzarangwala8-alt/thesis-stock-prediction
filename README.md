# Do Machine Learning Models Improve Stock Return Prediction?
### Evidence from S&P 500 Constituent Markets and Dimension Sensitivity (2015–2024)

[![Thesis Document](https://img.shields.io/badge/Thesis-PDF_Document-0A2240?style=for-the-badge&logo=adobeacrobatreader)](thesis.pdf)
[![LaTeX Source](https://img.shields.io/badge/LaTeX-Source_Code-008080?style=for-the-badge&logo=latex)](thesis/thesis.tex)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> **Master's Thesis in Economics and Data Analysis**  
> **Institution**: University of Verona, Department of Economics  
> **Candidate**: Murtuza Yusuf Rangwala (Matricola: VR508566)  
> **Supervisor**: Prof. Giuseppina Chesini  
> **Academic Year**: 2024/2025  

---

## 📌 Executive Summary

This repository contains the complete empirical code, data pipelines, model implementations, backtesting engines, and LaTeX manuscript source for the Master's thesis **"Do Machine Learning Models Improve Stock Return Prediction?"**.

The research investigates whether non-linear machine learning architectures and multi-modal sequence encoders yield statistically significant predictive outperformance over classical linear asset pricing benchmarks across S&P 500 constituent equities (2015–2024, 2,516 daily cross-sections).

### 🏆 Key Empirical Highlights

- **Predictive Superiority**: The proposed **Temporal Fusion Deep Multimodal Gated Attention (TFDMGA)** network achieves top out-of-sample accuracy across 5 expanding-window walk-forward folds (2020–2024):
  - **Daily Information Coefficient (IC)**: **`+0.0348`**
  - **Information Ratio (ICIR)**: **`3.12`** ($t$-statistic)
  - **ROC AUC**: **`0.6120`**
  - **Directional Accuracy**: **`56.84%`**
  - **Diebold-Mariano Test**: **`DM = 2.41`** ($p = 0.016$) over baseline LSTM models.
- **\$1,000 USD Account Compounding**: Under 10 bps transaction fees and 1-day execution delays, an initial **\$1,000 USD deposit** grows to:
  - **`$3,120.99 USD`** for baseline PyTorch LSTM ($Q_5$ Long-Only).
  - **`$4,368.50 USD`** for LSTM + 2:1 Take-Profit/Stop-Loss (TPSL) Risk Overlay.
  - **`$6,482.10 USD`** for **TFDMGA + 2:1 TPSL Risk Overlay**.
- **Factor Spanning & Market Efficiency**: Fama-French 5-factor spanning regressions yield an estimated net strategy alpha of **`α̂ = -0.18%`** ($p = 0.976$, $R^2 = 41.2\%$). The absence of unpriced excess alpha confirms that machine learning models operate as **dynamic factor allocation engines** rather than discoverers of unpriced market arbitrage.

---

## 🏗️ Repository Architecture

```
thesis final 2.0/
├── thesis.pdf                         # Final Compiled Master's Thesis Document (PDF)
├── thesis.docx                        # Final Editable Master's Thesis Document (Word)
├── README.md                          # Repository Documentation & Overview
├── requirements.txt                   # Project Python Dependencies
├── main.py                            # Master Pipeline Orchestration Entry Point
│
├── TFDMGA/                            # Production Custom Neural Network Package
│   ├── config.py                      # Model Hyperparameters & Walk-Forward Configuration
│   ├── model.py                       # Main TFDMGA Architecture Topology
│   ├── encoders.py                    # Causal 1D Dilated TCN Encoders
│   ├── attention.py                   # Sequential Directed Ring Attention Cascade
│   ├── fusion.py                      # 3-Way Macro Dynamic Gating & Transformer Fusion
│   ├── losses.py                      # Multi-Task Loss (Huber + Ranking + IC Loss)
│   ├── dataset.py                     # Sequence Lookback Window Dataset Construction
│   ├── train.py                       # AMP Mixed Precision Training Loop
│   └── optuna_search.py               # Optuna Hyperparameter Optimization Search
│
├── src/                               # Econometric & Machine Learning Modules
│   ├── data_pipeline.py               # Bloomberg Point-in-Time Data Ingestion Pipeline
│   ├── features.py                    # 59-Variable Feature Extraction & Rank Normalization
│   ├── baseline_models.py             # Fama-MacBeth 2-Pass OLS & Newey-West HAC SEs
│   ├── ml_models.py                   # LASSO, ElasticNet, Random Forest, XGBoost Ensembles
│   ├── lstm_model.py                  # PyTorch Multi-Layer LSTM Baseline Encoder
│   ├── backtest.py                    # Portfolio Decile Sorting & 10 bps Cost Engine
│   └── interpretability.py            # SHAP (SHapley Additive exPlanations) Analysis
│
├── thesis/                            # Complete LaTeX Manuscript Source Files
│   ├── thesis.tex                     # Master LaTeX Main File
│   ├── references.bib                 # BibTeX Citation Database (40+ Top-Tier References)
│   ├── figures/                       # High-Resolution Figures & Amiri Vector Fonts
│   │   ├── arabic_text_font.pdf       # Calligraphic Arabic Font Artwork
│   │   ├── equity_curves_comparison_21d.png
│   │   ├── rolling_sharpe_21d_feattech_fund.png
│   │   ├── shap_bar_plot_21d_feattech_fund.png
│   │   ├── shap_summary_plot_21d_feattech_fund.png
│   │   └── stop_loss_impact_curves.png
│   └── chapters/                      # Consolidated 5 Chapter TeX Files
│       ├── introduction.tex           # Chapter 1: Introduction & Research Questions
│       ├── literature_review.tex      # Chapter 2: Literature Review & Theoretical Foundations
│       ├── data_and_methodology.tex   # Chapter 3: Data Pipeline, Features & Architectures
│       ├── empirical_results.tex      # Chapter 4: Empirical Results, SHAP & Backtesting
│       └── conclusion.tex             # Chapter 5: Conclusion & Future Research Extensions
│
├── data/                              # Data Storage & Feature Mappings
│   └── processed/
│       └── selected_features.json     # Selected 53 ML Features & Group Mappings
│
└── scripts/                           # Execution & Validation Helper Scripts
    ├── select_features.py             # 3-Stage Feature Selection Pipeline
    ├── inspect_models.py              # Model Checkpoint Verification Script
    ├── validate_data.py               # Point-in-Time Data Integrity Validator
    ├── make_pdf_report.py             # Performance Summary Generator
    └── test_ff5_betas.py              # Fama-French 5-Factor Beta Verification
```

---

## 🧠 TFDMGA Architecture Deep Dive

The **Temporal Fusion Deep Multimodal Gated Attention (TFDMGA)** architecture combines four key deep learning innovations:

```
[Raw Features] ──> [Causal TCN Encoders] ──> [Directed Ring Attention] ──> [Macro Dynamic Gating] ──> [Transformer Fusion] ──> [Multi-Task Heads]
 (53 Inputs)         (Receptive Field = 15d)   (Tech < Sent < Fund < Macro)     (Temperature = 0.50)       (Pre-LN Stack)           (1d / 21d / 126d)
```

1. **Causal 1D Dilated TCN Encoders**: Stacked convolutions ($K=3, d \in \{1, 2, 4\}$) providing a 15-day receptive field without lookahead leakage ($R = 1 + (K-1)(2^L - 1)$).
2. **Sequential Directed Ring Attention Cascade**: Cross-modal interaction ($\text{Technical} \leftarrow \text{Sentiment} \leftarrow \text{Fundamental} \leftarrow \text{Macro}$) eliminating $O(M^2)$ parameter explosion.
3. **3-Way Macro Dynamic Gating**: Macro-conditioned trust weights ($w_{\text{tech}}, w_{\text{fund}}, w_{\text{sent}}$) scaled via temperature softmax ($\tau = 0.50$).
4. **Multi-Task Loss Function**: Composite optimization balancing point error, rank order, and correlation:
$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\operatorname{Huber}}(\delta=1.0) + 0.50 \cdot \mathcal{L}_{\text{rank}}(\gamma=0.10) + 1.0 \cdot \mathcal{L}_{\operatorname{IC}}$$

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/thesis-stock-prediction.git
cd thesis-stock-prediction
pip install -r requirements.txt
```

### 2. Run Data Ingestion & Feature Selection

Execute the 3-stage feature selection pipeline (missing values filter, $|\rho| < 0.85$ correlation filter, and Random Forest feature ranking):

```bash
python scripts/select_features.py
```

### 3. Train Baseline & Deep Learning Models

Run the main orchestration pipeline to execute Fama-MacBeth OLS, ElasticNet, Random Forest, XGBoost, PyTorch LSTM, and TFDMGA across 5 walk-forward test folds:

```bash
python main.py
```

To run Optuna hyperparameter optimization for TFDMGA:

```bash
python -m TFDMGA.optuna_search
```

### 4. Compile the LaTeX Manuscript

To compile `thesis.pdf` directly from LaTeX source:

```bash
cd thesis
pdflatex thesis.tex
bibtex thesis
pdflatex thesis.tex
pdflatex thesis.tex
```

---

## 📊 Summary of Baseline vs. Deep Learning Results

| Model Architecture | Feature Panel | Out-of-Sample Accuracy | ROC AUC | Daily IC | ICIR ($t$-stat) | Net Sharpe (10 bps) | \$1,000 USD Growth |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S&P 500 Buy & Hold** | Market Index | --- | --- | --- | --- | 0.52 | \$2,060.43 |
| **LASSO (Logistic $L_1$)** | Technical (16) | 54.18% | 0.5213 | +0.0246 | 2.67 | 0.48 | \$1,995.56 |
| **ElasticNet ($L_1+L_2$)** | Technical (16) | 54.04% | 0.5206 | +0.0237 | 2.51 | 0.48 | \$1,995.56 |
| **Random Forest** | Tech+Fund (46) | 54.76% | 0.5415 | +0.0332 | 3.00 | 0.74 | \$2,432.77 |
| **XGBoost** | Tech+Fund (46) | 54.37% | 0.5373 | +0.0247 | 3.02 | 0.72 | \$2,405.46 |
| **PyTorch LSTM** | Tech+Fund (46) | 56.12% | 0.6057 | +0.0298 | 2.85 | 1.15 | \$3,120.99 |
| **LSTM + 2:1 TPSL** | Tech+Fund (46) | 56.12% | 0.6057 | +0.0298 | 2.85 | 1.58 | \$4,368.50 |
| **TFDMGA + 2:1 TPSL** | **Master Panel (53)** | **56.84%** | **0.6120** | **+0.0348** | **3.12** | **2.14** | **\$6,482.10** |

---

## 📜 Citation

If you find this codebase or thesis research useful in your work, please cite it as follows:

```bibtex
@mastersthesis{rangwala2025machine,
  author       = {Murtuza Yusuf Rangwala},
  title        = {Do Machine Learning Models Improve Stock Return Prediction? Evidence from S\&P 500 Constituent Markets and Dimension Sensitivity (2015--2024)},
  school       = {University of Verona, Department of Economics},
  year         = {2025},
  type         = {Master's Thesis},
  supervisor   = {Prof. Giuseppina Chesini},
  address      = {Verona, Italy}
}
```

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).
