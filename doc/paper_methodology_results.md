## 3. Methodology

### 3.1 Problem Formulation

Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ be a heterogeneous financial graph where each node $v_i \in \mathcal{V}$ represents a publicly traded asset. We model market activity as a continuous stream of typed events $\{(s, r, t, \mathbf{f}_e)\}$, where $s, r \in \mathcal{V}$ are the source and (optional) target nodes, $t \in \mathbb{R}^+$ is a continuous timestamp, and $\mathbf{f}_e \in \mathbb{R}^{d_e}$ is an event feature vector.

The learning objective is next-day correlation prediction: given all events up to and including day $t$, predict the pairwise Pearson correlation $\rho_{ij}^{t+1}$ for all asset pairs $(i, j)$ on day $t+1$. This formulation is causally valid by construction — no label information from day $t+1$ is visible at inference time.

### 3.2 Dataset

**Universe.** We use a universe of $N = 30$ large-cap S&P 500 constituents selected by liquidity and chosen to cover all 11 GICS sectors: Information Technology (5), Financials (4), Health Care (3), Consumer Discretionary (3), Consumer Staples (2), Energy (2), Industrials (3), Communication Services (3), Materials (2), Utilities (2), and Real Estate (1).

**Period.** Daily data from 2020-01-01 to 2024-12-31 (5 years, 1,278 trading days). Adjusted closing prices are sourced from Yahoo Finance via the `yfinance` library.

**Event stream.** After parsing all data sources, the full event stream consists of **350,656 events** spanning 7 distinct event types (Table 1). Each event carries a 3-dimensional feature vector $\mathbf{f}_e = [\text{magnitude}, \text{direction}, \text{relative\_size}]$.

| Event Type | Description | Source |
|---|---|---|
| `PRICE_UPDATE` | Daily log-return signal | Yahoo Finance |
| `EARNINGS_REPORT` | Earnings release date | Yahoo Finance |
| `CORP_ACTION` | Dividend / stock split | Yahoo Finance |
| `CORRELATION_UPDATE` | DCC-GARCH pairwise $\rho$ crossing threshold | Estimated (§3.3) |
| `MACRO_RELEASE` | Macro surprise ($>1.5\sigma$) | FRED API |
| `FED_DECISION` | Federal funds rate change | FRED API |
| `CREDIT_DOWNGRADE` | OAS spread shock | FRED API |

**Table 1.** Event types in the DyFO financial event stream.

### 3.3 Correlation Target (DCC-GARCH)

Correlation labels are estimated via the Dynamic Conditional Correlation GARCH(1,1) model of Engle (2002), fitted over a rolling 252-day window. Let $\boldsymbol{\epsilon}_t$ be the vector of standardised GARCH(1,1) residuals. The conditional correlation matrix follows:

$$Q_t = (1 - a - b)\,\bar{Q} + a\,\boldsymbol{\epsilon}_{t-1}\boldsymbol{\epsilon}_{t-1}^\top + b\,Q_{t-1}$$

$$R_t = \mathrm{diag}(Q_t)^{-1/2}\,Q_t\,\mathrm{diag}(Q_t)^{-1/2}$$

where $\bar{Q}$ is the unconditional covariance matrix estimated over the training window. DCC-GARCH is preferred over rolling Pearson because it captures volatility clustering and provides time-varying second-moment estimates robust to heteroskedastic equity returns.

For the event stream, `CORRELATION_UPDATE` events are emitted whenever $|\rho_{ij}^t| \geq 0.3$ (sparsification threshold). Regression labels use the unsparsified DCC series to preserve continuous signal for all pairs.

### 3.4 Node Features

Each asset $v_i$ is described by a 20-dimensional time-varying feature vector $\mathbf{v}_i(t)$:

$$\mathbf{v}_i(t) = \left[\underbrace{r_{21d},\; \sigma_{21d},\; \beta_{63d}}_{\text{market signals}}\;,\;\underbrace{\mathbf{s}_i}_{\text{GICS one-hot, 11d}}\;,\;\underbrace{\tilde{m}_i}_{\text{log mcap}}\;,\;\underbrace{dd_i(t)}_{\text{drawdown}}\;,\;\underbrace{\boldsymbol{\pi}_i(t)}_{\text{regime probs, 3d}}\;,\;\underbrace{\tilde{v}_i(t)}_{\text{vol/avg}}\right]$$

where $r_{21d}$ is the 21-day log-return, $\sigma_{21d}$ the 21-day realised volatility, $\beta_{63d}$ the rolling 63-day market beta, $\mathbf{s}_i$ the GICS sector one-hot encoding, $\tilde{m}_i$ the cross-sectionally normalised log market cap, $dd_i(t)$ the current drawdown from peak, $\boldsymbol{\pi}_i(t)$ the regime probability vector from a hidden Markov model (zero-filled in the absence of an RDM module), and $\tilde{v}_i(t)$ the volume normalised by its 21-day mean.

### 3.5 Graph Structure and Edge Types

The static graph $\mathcal{G}$ contains four types of edges:

| Type | Description | Construction |
|---|---|---|
| `CORR` | Dynamic pairwise correlation | DCC-GARCH $|\rho| \geq 0.3$ |
| `SECT` | Same GICS sector | Binary indicator |
| `SUPL` | Supply-chain proximity | External mapping |
| `FACT` | Fama-French 5-factor co-loading | FF5 loading cosine $\geq 0.5$ |

All edge types are embedded into a shared $d_\text{edge} = 16$-dimensional space via a learned `Embedding` table and used as edge attributes in the Temporal Graph Attention layer.

### 3.6 TGN Architecture (DyFO Next-Day Variant)

DyFO builds on Temporal Graph Networks (Rossi et al., 2020), adapted to the next-day prediction setting. The full pipeline per trading day $t$ is:

**Memory.** Each node maintains a persistent memory vector $\mathbf{s}_i(t) \in \mathbb{R}^{d_m}$ with $d_m = 172$. Memory is inherited across train/validation/test splits (no reset at split boundaries) to model long-range temporal dependencies.

**Message Function.** For each event $(i, j, t, \mathbf{f}_e)$, a raw message for the source node is:

$$\mathbf{m}_i(t) = \left[\mathbf{s}_i(t^-) \;\|\; \mathbf{s}_j(t^-) \;\|\; \phi(\Delta t_i) \;\|\; \mathbf{f}_e \;\|\; \mathbf{e}_{\text{edge}} \;\|\; \mathbf{e}_{\text{event}}\right] \in \mathbb{R}^{479}$$

where $\phi(\Delta t) \in \mathbb{R}^{100}$ is the Time2Vec encoding (Kazemi et al., 2019) of the elapsed time since node $i$'s last event, and $\mathbf{s}_j(t^-)$ is replaced with **0** for node-only events (target $= -1$). A symmetric message is computed for the target node $j$ using $(\mathbf{s}_j, \mathbf{s}_i)$ as the primary/secondary memory pair.

**Unified Aggregation.** Source and target messages are pooled into a single list and passed to **one** `mean` aggregator call. This ensures that nodes appearing as both source and target within the same daily batch (common for correlation events) receive a single coherent aggregated message rather than an arbitrary sum of two independently aggregated tensors — a correctness fix relative to the vanilla TGN implementation.

**Memory Update.** The aggregated message $\bar{\mathbf{m}}_i$ is used to update memory via a GRU cell:

$$\mathbf{s}_i(t) = \mathrm{GRU}(\bar{\mathbf{m}}_i,\; \mathbf{s}_i(t^-))$$

Only nodes that received at least one message in day $t$ are updated; all other memories are carried forward unchanged. Memory is detached from the computation graph after each day (single-step TBPTT), preventing gradient accumulation across days.

**Temporal Graph Attention Embedding.** Node embeddings are computed via a single-layer multi-head attention (2 heads) over the static graph neighbourhood:

$$\mathbf{z}_i(t) = \mathrm{MLP}\!\left(\mathbf{h}_i \,\|\, \mathrm{MultiHeadAttn}\!\left(\mathbf{h}_i,\;\{\mathbf{h}_j \,\|\, \mathbf{f}_{ij} \,\|\, \phi(\Delta t_{ij})\}_{j \in \mathcal{N}(i)}\right)\right)$$

where $\mathbf{h}_i = [\mathbf{s}_i(t) \,\|\, \mathbf{v}_i(t)] \in \mathbb{R}^{192}$ and $\mathbf{z}_i(t) \in \mathbb{R}^{100}$.

**Decoder.** A 3-layer MLP with architecture $[200 \to 64 \to 32 \to 1]$ takes the concatenated pair embedding $[\mathbf{z}_i \,\|\, \mathbf{z}_j] \in \mathbb{R}^{200}$ and outputs $\hat{\rho}_{ij}^{t+1} \in [-1, 1]$ via a $\tanh$ final activation.

**Weight Initialisation.** The GRU recurrent weights ($W_{ih}$, $W_{hh}$) are initialised with orthogonal matrices (Saxe et al., 2013) to preserve gradient norms at epoch 1. All linear layers use Xavier uniform initialisation.

The total parameter count is **556,909**.

### 3.7 Training Protocol

**Pre-training objective.** The model is trained to minimise the Huber loss (SmoothL1, $\delta = 1$) between predicted and DCC-GARCH correlations over all known pairs on day $t+1$:

$$\mathcal{L} = \frac{1}{|\mathcal{P}_t|} \sum_{(i,j) \in \mathcal{P}_t} \ell_\delta\!\left(\hat{\rho}_{ij}^{t+1},\, \rho_{ij}^{t+1}\right)$$

Huber loss is chosen for robustness to outlier correlations near $\pm 1$ that arise during market stress periods.

**Walk-forward split.** The 1,278 trading days are partitioned chronologically into train (60%, 766 days), validation (20%, 256 days), and test (20%, 256 days). Memory state is **not** reset between splits; validation and test proceed with the memory inherited from the preceding period, reflecting the intended deployment setting.

**Optimisation.** Adam optimiser with learning rate $\eta = 2 \times 10^{-4}$, weight decay $\lambda = 10^{-4}$. A linear learning rate warmup over the first 2 epochs ramps $\eta$ from $10^{-4}$ to $2 \times 10^{-4}$, reducing gradient magnitude during the first pass through the training data when memory is zero-initialised. Gradients are clipped to $\ell_2$-norm $\leq 0.5$. Early stopping with patience 5 monitors validation $R^2$, and the best checkpoint is restored for test evaluation.

**Reproducibility.** All experiments are run with 5 independent random seeds $\{42, 43, 44, 45, 46\}$. Data preparation (downloads, DCC-GARCH estimation) is performed once and shared across seeds to isolate stochasticity to model initialisation and dropout.

---

## 4. Experiments

### 4.1 Evaluation Metrics

We report the following metrics computed on the held-out test set (last 256 trading days):

- **R²** (coefficient of determination): fraction of variance in $\rho_{ij}^{t+1}$ explained by $\hat{\rho}_{ij}^{t+1}$.
- **Spearman $\rho$**: rank correlation between predicted and actual correlations, measuring ordinal accuracy of the predicted correlation structure.
- **MAE**: mean absolute error in correlation units.

Additionally, we report derived classification metrics (Precision, Recall, F1 at $|\hat{\rho}| \geq 0.5$) that reflect the practical accuracy of identifying high-correlation asset pairs — the primary use case for portfolio construction.

### 4.2 Main Results

Table 2 reports per-seed test performance across 5 independent runs. All 5 seeds converged to high-quality solutions, demonstrating that the orthogonal GRU initialisation, LR warmup, and gradient clipping effectively eliminate training instability.

| Seed | R² | Spearman $\rho$ | MAE | cls-F1 |
|:---:|:---:|:---:|:---:|:---:|
| 42 | 0.756 | 0.917 | 0.055 | 0.686 |
| 43 | 0.766 | 0.927 | 0.054 | 0.679 |
| 44 | **0.812** | **0.936** | **0.048** | **0.727** |
| 45 | 0.799 | 0.932 | 0.050 | 0.768 |
| 46 | 0.768 | 0.926 | 0.055 | 0.707 |
| **Mean ± Std** | **0.780 ± 0.024** | **0.928 ± 0.007** | **0.052 ± 0.003** | **0.713 ± 0.034** |

**Table 2.** Test set performance of DyFO across 5 random seeds. Walk-forward protocol: train 2020–2022 (766 days), val 2022–2023 (256 days), test 2023–2024 (256 days). All metrics are averaged over all  $\binom{30}{2} = 435$ asset pairs per test day.

The model achieves **R² = 0.780 ± 0.024** and **Spearman $\rho$ = 0.928 ± 0.007** on the test period, demonstrating that temporal graph representations capture economically meaningful co-movement structure. The low variance across seeds (σ(R²) = 0.024, σ(Spearman) = 0.007) confirms robust convergence.

The rank correlation of 0.928 is particularly relevant for practical applications: asset managers primarily care about the *ordering* of correlation pairs (to select diversifying or clustering positions), and a Spearman of 0.928 indicates that DyFO's embeddings faithfully preserve the full correlation rank structure one day ahead.

### 4.3 Training Stability

Previous experiments with default Adam learning rate ($\eta = 10^{-3}$) and no gradient clipping exhibited catastrophic divergence at epoch 2 for 2 of 5 seeds (R² reaching $-22.7$). The instability mechanism was identified as explosive gradients through the GRU input gate when zero-initialised memory is first processed with large weight norms.

Three targeted fixes resolved the issue completely:

1. **Orthogonal GRU initialisation** — $W_{ih}, W_{hh}$ initialised to orthogonal matrices, keeping singular values at 1.0 and bounding the spectral norm of the first gradient pass.
2. **Linear LR warmup (2 epochs)** — $\eta$ scaled from $0.5\eta_\text{max}$ to $\eta_\text{max}$, halving the update magnitude during the critical memory-cold-start phase.
3. **Gradient clipping ($\ell_2 \leq 0.5$)** — stricter than the conventional 1.0 clip, providing a final safeguard against rare large-gradient events at high-volatility days.

These modifications did not reduce best-case performance; seed 44 achieved the top result (R² = 0.812) in the stabilised configuration.
