# PhD Research Proposal

## Title: Dynamic Neural Asset Pricing: Bridging Deep Learning, Alternative Data, and Macroeconomic State Spaces in High-Dimensional Empirical Finance

---

### Abstract
This research proposal outlines a unified **Dynamic Neural Asset Pricing** framework designed to address critical structural and empirical limitations of traditional asset pricing models and standard machine learning (ML) architectures. Although ML models (such as tree ensembles and transformers) have demonstrated predictive superiority over linear benchmarks (like the Fama-French 5-Factor model), they frequently suffer from major limitations in financial applications. These include: ignoring non-stationary macroeconomic regimes, failing to fuse multi-modal unstructured textual streams, neglecting non-linear transaction cost dynamics (such as market impact and liquidity constraints), and relying on path-dependent heuristic execution rules (such as ad-hoc Take-Profit/Stop-Loss caps).

To resolve these challenges, this PhD project proposes a three-year research agenda structured around four integrated work packages:
1. **Multi-Modal Spatio-Temporal Graph Neural Networks (GNNs)**: Discarding arbitrary 2D grid layouts to model the cross-section of stocks as an economic graph with dynamic edge attention, integrating temporal attention (Temporal Fusion Transformers) with exponential time-decay sentiment embeddings from financial LLMs.
2. **Macroeconomic-Conditioned Attention via Hyper-LoRA**: Developing a Conditional Transformer where attention projection matrices are dynamically modulated by a lookahead-free Point-in-Time (PIT) macroeconomic state vector using Lipschitz-regularized Hypernetworks generating low-rank adapters.
3. **Deep Reinforcement Learning (DRL) for Frictional Portfolio Execution**: Training a continuous-action DRL agent under a differentiable Quadratic Programming (QP) solver layer or Dirichlet policy constraints, optimized using a recursive Differential Sharpe Ratio reward function that incorporates non-linear market impact (the square-root law).
4. **Conditional Variational Autoencoder (CVAE) Factor Models**: Formulating a probabilistic generative latent factor framework that projects stock returns onto conditional factor spaces, using Explainable AI (XAI) tools (Integrated Gradients and SHAP) to map latent factor loadings back to economic characteristics.

---

## 1. Introduction and Academic Background
Empirical asset pricing seeks to identify the drivers of expected stock returns. In standard contemporaneous asset pricing frameworks (e.g., Fama and French, 2015), the cross-section of excess returns is modeled as:
\[R_{i, t+1} - R_{f, t+1} = \alpha_i + \beta_{i, M} (R_{M, t+1} - R_{f, t+1}) + \beta_{i, SMB} SMB_{t+1} + \beta_{i, HML} HML_{t+1} + \beta_{i, RMW} RMW_{t+1} + \beta_{i, CMA} CMA_{t+1} + \epsilon_{i, t+1}\]
Where the LHS excess return and RHS factor returns are contemporaneous. For predictive asset pricing, expected returns are modeled as functions of lagged, firm-specific characteristics \(Z_{i, t}\):
\[R_{i, t+1} - R_{f, t+1} = g(Z_{i, t}) + \epsilon_{i, t+1}\]
Where \(g(\cdot)\) is a mapping function. Traditional finance restricts \(g(\cdot)\) to be linear, whereas recent literature (Gu, Kelly, and Xiu, 2020) leverages deep learning to model the non-linear relationship.

In our preliminary work forecasting S&P 500 daily returns (2015–2024), we compared regularized linear baselines, non-linear tree ensembles (XGBoost), and a Tabular Transformer (similar to FT-Transformer; Gorishniy et al., 2021) across three feature dimensions (11, 50, and 80 features from Bloomberg Professional fundamentals and technical indicators). Over a 21-day horizon, the 80-feature Tabular Transformer Classifier achieved an out-of-sample Sharpe ratio of 0.7651 (annualized excess return of 18.15\%), outperforming the passive S&P 500 benchmark (Sharpe = 0.7103). Fama-French 5-Factor spanning regressions of the optimal long portfolio:
\[R_{p, t+1} - R_{f, t+1} = \alpha + \beta_1 (R_{M, t+1} - R_{f, t+1}) + \beta_2 SMB_{t+1} + \beta_3 HML_{t+1} + \beta_4 RMW_{t+1} + \beta_5 CMA_{t+1} + \eta_{t+1}\]
revealed no statistically significant alpha (\(\alpha_{\text{FF}} = 2.10\%\) annualized, \(t\text{-stat} = 0.62\)) and a high adjusted \(R^2\) of 89.31\%. This demonstrates that the deep learning model's out-of-sample performance is primarily driven by dynamic exposures to systematic risk factors rather than idiosyncratic arbitrage. 

Furthermore, while applying a daily 2:1 Take-Profit/Stop-Loss (TP/SL) exit rule (+4\% / -2\% caps) raised the out-of-sample Sharpe ratio to the 2.7–3.5 range, such daily rules are path-dependent, suffer from lookahead bias under daily-bar aggregation, and ignore execution slippage and transaction costs. At a daily rebalancing frequency, even a 5 basis point (bps) execution fee would erase these returns, illustrating the critical need to model non-linear frictions and optimal stochastic control in predictive models.

---

## 2. Research Gaps
1. **Spatial Representation Gaps in Equity Cross-Sections**: Standard machine learning models treat stocks as independent tabular rows, failing to model the economic network structure. Previous attempts to arrange stocks in a 2D matrix for CNN processing violate spatial translation invariance, since stock ordering is arbitrary. An economically structured graph representation is needed to model cross-sectional spillover effects.
2. **Overfitting to Macroeconomic Regime Shifts**: Standard neural pricing models apply static weights across time. Although conditioning attention layers on macroeconomic state variables can handle temporal non-seasonarity, generating high-dimensional projection matrices from low-frequency macroeconomic vectors introduces parameter explosion and severe overfitting. A parameter-efficient conditioning mechanism is required.
3. **The Disconnect Between Prediction and Execution**: Standard models output statistical forecasts (e.g., sign or magnitude of returns) which are mapped to portfolios using heuristic weighting or ad-hoc exit rules. These heuristics do not optimize utility in the presence of non-linear market impact (the square-root law of transaction costs) and capacity constraints.
4. **Latent Non-Linear Factor Interpretability**: Spanning regressions indicate that machine learning portfolios capture systematic risk. However, standard linear factor baselines cannot capture the non-linear risk factors that neural networks exploit. We need generative non-linear factor models that remain economically interpretable.

---

## 3. Methodology and Research Objectives

### WP1: Multi-Modal Spatio-Temporal Graph Neural Networks (GNNs)
To capture spatial (cross-sectional) and temporal dependencies without arbitrary matrix layouts, we propose a Spatio-Temporal Graph Neural Network:
*   **Economic Graph Structure**: We define the stock universe as a graph \(\mathcal{G}_t = (\mathcal{V}, \mathcal{E}_t)\). Nodes \(\mathcal{V}\) represent stocks, and edges \(\mathcal{E}_t\) represent relationships derived from supply-chain linkages, GICS sector classification, and dynamic correlation matrices.
*   **Temporal-First Spatial-Second Aggregation**: To model sequence dynamics, we run a Temporal Fusion Transformer (TFT; Lim et al., 2021) at the individual asset level first. This compresses the historical time-series of features into a temporal node embedding \(H_{i, t}\):
    \[H_{i, t} = \text{TFT}(X_{i, 1:t})\]
    Second, these temporal node embeddings are passed through a Graph Attention Network (GAT) to aggregate spatial spillovers:
    \[Z_{i, t} = \text{GAT}(H_t, \mathcal{E}_t)_i\]
*   **Dynamic Graph Regularization**: To handle time-varying correlation graphs without introducing topological noise, we parameterize the adjacency matrix using self-attention over node features:
    \[A_{i,j,t} = \text{Softmax}\left( \frac{\Phi(H_{i,t}) \Psi(H_{j,t})^T}{\sqrt{d_k}} \right)\]
    where \(\Phi\) and \(\Psi\) are projection matrices, enabling the graph structure to be learned end-to-end.
*   **Asynchronous Textual Fusion and Sentiment Decay**: Sentiment scores \(S_{i, t_k}\) from FinBERT (generated from news and SEC filings published at asynchronous times \(t_k\)) are aligned with daily features using an exponential decay aggregator:
    \[E_{i,t} = \sum_{t_k \le t} e^{-\lambda (t - t_k)} S_{i, t_k}\]
    where \(\lambda > 0\) is a half-life parameter (e.g., 5-day news decay). This embedding \(E_{i,t}\) is concatenated with the tabular feature panel before temporal processing.

### WP2: Macroeconomic-Conditioned Attention via Hyper-LoRA
To adapt the network to macroeconomic regime shifts without overfitting, we condition the multi-head attention projection matrices on a lookahead-free Point-in-Time (PIT) macro vector:
*   **Point-in-Time Macro Vector**:
    \[M_t^{\text{PIT}} = \left[ VIX_t, \text{Spread}_t, \text{CPI}_{t - \delta(t)} \right]^T\]
    where \(\delta(t)\) represents the variable publication release lag (e.g., CPI has a 10-15 day lag), preventing lookahead bias.
*   **Hypernetwork-generated Low-Rank Adaptation (Hyper-LoRA)**:
    We parameterize the projection matrices as:
    \[W_Q(M_t^{\text{PIT}}) = W_{Q, 0} + A_Q(M_t^{\text{PIT}}) B_Q(M_t^{\text{PIT}})\]
    where \(W_{Q, 0} \in \mathbb{R}^{d \times d_k}\) represents the base, static attention weights, and \(A_Q(M_t^{\text{PIT}}) \in \mathbb{R}^{d \times r}\), \(B_Q(M_t^{\text{PIT}}) \in \mathbb{R}^{r \times d_k}\) are low-rank matrices (\(r \ll d\)) output by an auxiliary Multi-Layer Perceptron (MLP) hypernetwork.
*   **Lipschitz Regularization**: To prevent weight explosions and stabilize training, we apply spectral normalization to the MLP layers:
    \[\text{Lip}(W_Q(M_t^{\text{PIT}})) \le L\]
    ensuring that small macroeconomic transitions do not cause chaotic jumps in the attention weights.

### WP3: Deep Reinforcement Learning for Frictional Portfolio Control
To replace heuristic exit rules, we formulate portfolio construction as a stochastic optimal control problem solved via a continuous-action DRL agent (PPO):
*   **Differentiable QP Layer and Simplex Constraints**: To avoid gradient saturation and instabilities of Softmax (the Softmax mapping problem), the DRL policy outputs target characteristic exposures \(a_t\) or factor loadings. The actual stock weights \(w_t\) are resolved deterministically using a differentiable Quadratic Programming (QP) solver layer (e.g., cvxpylayers):
    \[\min_{w_t} \|w_t - w_{\text{benchmark}}\|^2 \quad \text{subject to} \quad C_t^T w_t = a_t, \ \mathbf{1}^T w_t = 1, \ w_t \ge 0\]
    where \(C_t\) is the stock characteristics matrix. Alternatively, policy actions are parameterized using a Dirichlet distribution to naturally satisfy simplex constraints.
*   **Recursive Differential Sharpe Ratio Reward**: To align the reward with economic utility theory and avoid penalizing positive return outliers, we implement the recursive Differential Sharpe Ratio (DSR; Moody & Saffell, 2001):
    \[R_t = \frac{B_{t-1} \Delta A_t - \frac{1}{2} A_{t-1} \Delta B_t}{\left(B_{t-1} - A_{t-1}^2\right)^{3/2}} - \text{TC}_{p,t}\]
    where \(A_t = \eta A_{t-1} + (1 - \eta) r_{p,t}\) and \(B_t = \eta B_{t-1} + (1 - \eta) r_{p,t}^2\) represent exponentially moving estimates of the first and second moments of portfolio returns, and \(\Delta A_t = A_t - A_{t-1}\), \(\Delta B_t = B_t - B_{t-1}\).
*   **Non-Linear Frictional Costs**: We incorporate institutional market impact using the square-root law:
    \[\text{TC}_{p, t} = \sum_{i \in \mathcal{V}} \left( c_{\text{spread}} |w_{i, t} - w_{i, t^-}| + \gamma_{\text{impact}} \sigma_{i, t} \left( \frac{\text{AUM} \cdot |w_{i, t} - w_{i, t^-}|}{\text{ADV}_{i, t}} \right)^{0.5} \right)\]
    Where \(\text{ADV}_{i, t}\) is the average daily volume, and \(\sigma_{i, t}\) is daily return volatility.

### WP4: Conditional Variational Autoencoder (CVAE) Factor Models
To bridge the gap between machine learning and economic theory, we formulate a probabilistic generative latent factor structure:
*   **Conditional Variational Autoencoder (CVAE)**:
    *   **Prior Distribution**: \(f_{t+1} \sim \mathcal{N}(0, I)\) (latent factors)
    *   **Recognition Encoder**: \(q(f_{t+1} | R_{t+1}, Z_t) = \mathcal{N}(\mu_{\phi}(R_{t+1}, Z_t), \text{diag}(\sigma^2_{\phi}(R_{t+1}, Z_t)))\)
    *   **Generative Decoder**: \(p_{\theta}(R_{t+1} | f_{t+1}, Z_t) = \mathcal{N}(h_{\theta}(Z_t)^T f_{t+1}, \sigma^2_{\epsilon} I)\)
    *   **Loss (ELBO)**:
        \[\mathcal{L}(\theta, \phi; R_{t+1}, Z_t) = \mathbb{E}_{q}\left[ \log p_{\theta}(R_{t+1} | f_{t+1}, Z_t) \right] - D_{\text{KL}}\left( q(f_{t+1} | R_{t+1}, Z_t) \,||\, p(f_{t+1}) \right)\]
    This probabilistic formulation allows for conditional simulations of factor returns, enabling robust stress-testing of portfolios.
*   **Explainable AI (XAI)**: We apply SHAP values and Integrated Gradients to the decoder network \(h_{\theta}(Z_t)\). This maps the marginal contribution of GNN embeddings and alternative features to the non-linear systematic factor loadings.

---

## 4. Expected Research Contributions
1. **Theoretical**: A mathematically rigorous, parameter-efficient Hyper-LoRA framework for conditioning attention layers on macro-states, proving that transformers can adapt to financial regime shifts without overfitting.
2. **Methodological**: Development of a multi-modal spatio-temporal GNN that combines structured fundamentals, technical indicators, and financial sentiment on a dynamically constructed economic graph.
3. **Empirical**: Demonstration of a DRL portfolio agent that outperforms traditional optimization models under realistic non-linear market impact, providing a robust alternative to heuristic TP/SL rules.
4. **Interpretability**: A unified approach to decompose neural asset pricing performance into latent systematic factors, validating the results against economic asset pricing theory.

---

## 5. Timeline and Milestones
*   **Month 01 - 12 (Year 1)**: Core coursework, database assembly (Bloomberg, SEC sentiment, macro-indices, GICS maps). Development, training, and validation of WP1 (Multi-Modal GNN-Transformer).
*   **Month 13 - 24 (Year 2)**: Implementation of WP2 (LoRA-Conditioned Attention). Development of WP3 (DRL agent with non-linear market impact). Submission of first paper to a leading quantitative finance journal.
*   **Month 25 - 36 (Year 3)**: Execution of WP4 (Latent Factor Autoencoder and XAI). Thesis compilation and defense.

---

## 6. Targeted Academic Journals and Conferences
*   **Journals**: *Review of Financial Studies*, *Journal of Financial and Quantitative Analysis*, *Journal of Financial Data Science*, *Quantitative Finance*.
*   **Conferences**: *International Conference on AI in Finance (ICAIF)*, *ACM International Conference on AI in Finance*.

---

## 7. Key References
*   Fama, E. F., and French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1-22.
*   Gorishniy, Y., Rubachev, I., Hrbanov, V., and Babenko, A. (2021). Revisiting Deep Learning Models for Tabular Data. *Advances in Neural Information Processing Systems (NeurIPS)*.
*   Gu, S., Kelly, B., and Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *The Review of Financial Studies*, 33(5), 2223-2273.
*   Gu, S., Kelly, B., and Xiu, D. (2021). Autoencoder Asset Pricing. *Journal of Econometrics*, 222(1), 429-450.
*   Kelly, B., Pruitt, S., and Su, Y. (2019). Instrumented Principal Component Analysis for Asset Pricing. *The Journal of Finance*, 74(3), 1387-1423.
*   Lim, B., Arık, S. Ö., Loeff, N., and Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748-1764.
*   Lopez-Lira, A., and Tang, Y. (2023). Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models. *Working Paper*.
*   Loughran, T., and McDonald, B. (2011). When is a Liability not a Liability? Textual Analysis, Dictionaries, and 10-Ks. *The Journal of Finance*, 66(1), 35-65.
*   Moody, J., and Saffell, M. (2001). Learning to trade via direct reinforcement. *IEEE Transactions on Neural Networks*, 12(4), 875-889.
