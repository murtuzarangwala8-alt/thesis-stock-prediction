# TFDMGA — Temporal Fusion Deep Multimodal Gated Attention Network

> **Master's Thesis**: *Do Machine Learning Models Improve Stock Return Prediction?  
> Evidence from Financial Markets*  
> Dataset: S&P 500 Constituents, 2014–2024

---

## Architecture

TFDMGA is a novel multi-modal deep learning architecture for joint prediction of
**1-day** and **21-day** forward stock returns from three heterogeneous signal types.

```
x_tech  (B, 46)  ──► TechnicalEncoder  ──► MHSA_tech  ──►┐
x_fund  (B, 192) ──► FundamentalEncoder ──► MHSA_fund  ──►├──► CrossModalAttention (ring)
x_macro (B, 26)  ──► MacroEncoder       ──► MHSA_macro ──►┘
                                                            │
                              DynamicGatingNetwork ◄────────┤
                                       │
                              ResidualFusionBlock
                                       │
                       TransformerEncoderBlock × N
                                       │
                            ┌──────────┴──────────┐
                        Head_1d (→ 1)         Head_21d (→ 1)
```

### Key innovations
| Component | Description |
|---|---|
| **Modal Encoders** | Independent LayerNorm + Residual Blocks + GELU per modality |
| **Per-Modal MHSA** | Flash Attention (SDPA) applied independently to each modality token |
| **Cross-Modal Attention** | Ring topology: Tech←Fund, Fund←Macro, Macro←Tech |
| **Dynamic Gating** | Temperature-scaled softmax learns per-sample modality trust weights |
| **Residual Fusion** | Gated combination + LayerNorm + FF + skip |
| **Transformer Stack** | N configurable Pre-LN transformer blocks on the fused representation |
| **Dual Heads** | Separate MLP heads for 1d and 21d predictions |

---

## Package Structure

```
TFDMGA/
├── __init__.py         Public API exports
├── config.py           TFDMGAConfig dataclass — all hyperparameters
├── attention.py        MHSA, CrossModalAttention, TransformerEncoderBlock
├── fusion.py           ModalEncoder, DynamicGatingNetwork, ResidualFusionBlock
├── model.py            Full TFDMGA architecture
├── losses.py           MultiTaskLoss, HuberLoss, RankingLoss, ICLoss
├── metrics.py          IC, RankIC, ICIR, Sharpe, Sortino, Calmar, MDD, turnover, costs
├── dataset.py          MasterDataStore (loads parquet once), FinancialPanelDataset
├── trainer.py          Training engine (AMP, fused AdamW, torch.compile, checkpointing)
├── walkforward.py      5-fold expanding-window validation engine
├── optuna_search.py    50-trial Optuna study (MedianPruner + TPE sampler)
├── evaluate.py         Ensemble evaluation + publication plots
├── train.py            CLI entry point
└── requirements.txt
```

---

## Installation (RunPod)

```bash
# 1. Upload the TFDMGA/ package to /workspace on your RunPod pod
# 2. Install dependencies (torch is pre-installed in RunPod image)
pip install optuna optuna-dashboard tensorboard matplotlib scipy pyarrow tqdm psutil

# 3. Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.__version__)"
```

---

## Usage

### Full pipeline (Optuna → Walk-Forward → Evaluation)
```bash
cd /workspace
python -m TFDMGA.train \
    --data_path /workspace/data/master_panel_features.parquet \
    --run_optuna \
    --n_optuna_trials 50 \
    --n_trial_epochs 30 \
    --start_fold 1 \
    --end_fold 5 \
    --max_epochs 150 \
    --seed 42
```

### Skip Optuna, train with defaults
```bash
python -m TFDMGA.train \
    --data_path /workspace/data/master_panel_features.parquet \
    --d_model 256 --n_heads 8 --n_transformer_blocks 4 \
    --batch_size 2048 --lr 3e-4
```

### Resume from fold 3
```bash
python -m TFDMGA.train \
    --data_path /workspace/data/master_panel_features.parquet \
    --start_fold 3 --end_fold 5
```

### Evaluation only (after all folds trained)
```bash
python -m TFDMGA.train \
    --data_path /workspace/data/master_panel_features.parquet \
    --skip_training
```

---

## GPU Optimisations

| Optimisation | Status |
|---|---|
| `torch.compile(mode="reduce-overhead")` | ✅ Enabled by default |
| AMP (bfloat16 / float16) | ✅ Enabled via `--use_amp` |
| Fused AdamW CUDA kernel | ✅ Auto-detected on CUDA 11.6+ |
| TF32 matrix math | ✅ Enabled on Ampere+ GPUs |
| Flash Attention 2 (SDPA) | ✅ Automatic via `F.scaled_dot_product_attention` |
| Dataset VRAM caching | ✅ Auto: uploads if dataset fits with 60% safety margin |
| Pinned memory + async DMA | ✅ Fallback when dataset is too large for VRAM |
| Persistent workers + prefetch | ✅ Configurable `--num_workers` / `--prefetch_factor` |
| Gradient clipping | ✅ `--grad_clip 1.0` |

---

## Walk-Forward Folds

| Fold | Train | Val |
|------|-------|-----|
| 1 | 2015–2018 | 2019 |
| 2 | 2015–2019 | 2020 |
| 3 | 2015–2020 | 2021 |
| 4 | 2015–2021 | 2022 |
| 5 | 2015–2022 | 2023 |
| **Test** | *never trained on* | **2024** |

---

## Metrics

| Category | Metrics |
|---|---|
| Statistical | MSE, MAE, R² |
| Signal | IC (Pearson), Rank-IC (Spearman), ICIR, Hit Ratio |
| Portfolio | Sharpe, Sortino, Calmar, Max Drawdown, Annual Return, Annual Vol |
| Decomposed | Long-only, Short-only, Long-Short |
| Costs | Turnover, Net Return at 5 / 10 / 20 bps round-trip |

---

## Output Files

After a full run:
```
/workspace/
├── checkpoints/TFDMGA/
│   ├── fold1_best.pt
│   ├── fold2_best.pt
│   └── ...
├── logs/TFDMGA/
│   ├── TFDMGA_train.log
│   ├── fold1_metrics.csv
│   └── tensorboard/fold1/
├── results/TFDMGA/
│   ├── walkforward_summary.csv
│   ├── final_evaluation_results.json
│   ├── test_predictions_ensemble.parquet
│   ├── optuna_trials.csv
│   ├── optuna_best_params.json
│   └── plots/
│       ├── cumulative_returns.png
│       ├── drawdown.png
│       ├── daily_ic_pred_1d.png
│       ├── gate_weights.png
│       └── prediction_distribution.png
```

---

## Citation

If you use this framework in your research:
```bibtex
@mastersthesis{tfdmga2024,
  title  = {Do Machine Learning Models Improve Stock Return Prediction?
             Evidence from Financial Markets},
  author = {[Author]},
  year   = {2024},
  school = {[University]},
  note   = {TFDMGA: Temporal Fusion Deep Multimodal Gated Attention Network}
}
```
