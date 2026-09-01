# Feature Specification: Neuro-Symbolic LLM, Continuous DRL & Discrete DQN Framework

**Feature ID**: `neurosymbolic-drl-dqn-advances`  
**Base Branch**: `dev` (synchronized at commit `3a49400`)  
**Status**: In Progress  
**Methodology**: Tech Lead's Club - Spec-Driven Development (TLC-SDD)  
**Execution Sequence**: (3) Neuro-Symbolic LLM/GraphRAG $\to$ (1) Continuous DRL $\to$ (2) Discrete DQN  

---

## 1. Overview & Doctoral Research Vision

Following the successful merge of the multi-model benchmark (10 models) and the practical portfolio demo, this specification establishes DyFO as an **advanced structural reasoning and perception engine** across three modern AI paradigms:

1. **(3) Neuro-Symbolic AI & LLMs (GraphRAG, Temporal Triples & Causal Reasoning)**:
   - Bridges DyFO's continuous temporal graph predictions with Large Language Models (LLMs) via causal GraphRAG, extracting typed relational subgraphs, generating natural language risk explanations, and compiling symbolic constraints for quadratic portfolio optimizers.
2. **(1) Continuous Deep Reinforcement Learning (Actor-Critic / PPO)**:
   - Leverages DyFO's dynamic graph embeddings ($\mathbf{z}_i(t) \in \mathbb{R}^{100}$) as relational state augmentation, breaking the *asset permutation symmetry* and action-space collapse ($1/N$ uniform degeneration) in continuous portfolio rebalancing.
3. **(2) Discrete Deep Q-Networks (Dueling Double-DQN with Prioritized Replay)**:
   - Formulates a discrete Markov Decision Process (MDP) for dynamic regime switching and tail-risk hedging conditioned on DyFO graph centrality and eigen-spectrum.

---

## 2. Requirements & Traceability

### Phase 1: Neuro-Symbolic AI & LLM GraphRAG (Sequence Step 3)

#### REQ-NS1: Temporal Subgraph Extraction & Semantic Triples Serialization
- **REQ-NS1.1**: Implement `CausalSubgraphExtractor` to filter the top-$k$ significant co-movement shocks ($\Delta\hat{\rho}_{ij}$) and relational paths (`SAME_SECTOR`, `SUPPLIER_TO`, `SUBSIDIARY_OF`, `FED_DECISION`, `EARNINGS`) around target assets.
- **REQ-NS1.2**: Serialize subgraphs into typed JSON-LD, RDF/Turtle, and Human-Readable Natural Text Triples for LLM ingestion.
- **REQ-NS1.3**: Ensure zero look-ahead bias: all subgraphs must contain strictly causal historical and $t+1$ forecasted edges.

#### REQ-NS2: Neuro-Symbolic Prompt Engine & Causal Explanation
- **REQ-NS2.1**: Implement `GraphRAGPromptEngine` supporting customizable reasoning templates for:
  1. *Causal Risk Attribution* (identifying why specific asset correlations spiked).
  2. *Macro Contagion Warning* (tracing spillover from macro nodes to sector clusters).
  3. *Symbolic Constraint Proposal* (generating mathematical allocation bounds).
- **REQ-NS2.2**: Support pluggable LLM backends (OpenAI API, Anthropic Claude, Google Gemini, and Local Mock/Ollama/vLLM) with graceful fallback and deterministic JSON output validation.

#### REQ-NS3: Symbolic Constrained Portfolio Optimization
- **REQ-NS3.1**: Implement `SymbolicConstraintParser` to validate and compile LLM output into linear inequality matrices ($A_{\text{sym}} \mathbf{w} \le \mathbf{b}_{\text{sym}}$).
- **REQ-NS3.2**: Integrate symbolic constraints into `DyFOAdapter` with Higham SPD covariance projection, solving:
  $$\min_{\mathbf{w}} \mathbf{w}^T \mathbf{\Sigma}_t \mathbf{w} \quad \text{s.t.} \quad \mathbf{1}^T \mathbf{w} = 1, \quad \mathbf{w} \ge 0, \quad A_{\text{sym}} \mathbf{w} \le \mathbf{b}_{\text{sym}}$$

#### REQ-NS4: Neuro-Symbolic Demo & Unit Verification
- **REQ-NS4.1**: Create `examples/demo_dyfo_llm_neurosymbolic.py` demonstrating end-to-end GraphRAG prompt extraction, reasoning, constraint generation, and backtested performance.
- **REQ-NS4.2**: Implement comprehensive unit test suite in `tests/test_neurosymbolic_graphrag.py`.

---

### Phase 2: Continuous Deep Reinforcement Learning (Sequence Step 1)

#### REQ-DRL1: Relational State Augmentation & Attention Policy
- **REQ-DRL1.1**: Build `GraphAugmentedDRLState` packaging node embeddings $\mathbf{Z}_t \in \mathbb{R}^{N \times 100}$, current portfolio weights $\mathbf{w}_{t-1}$, and macro regime probabilities $\boldsymbol{\pi}_t$.
- **REQ-DRL1.2**: Implement `RelationalActorCriticPolicy` using asset-to-asset cross-attention conditioned on DyFO topological features.

#### REQ-DRL2: Risk-Regularized Continuous Policy Optimization (PPO/A2C)
- **REQ-DRL2.1**: Formulate multi-objective reward function penalizing turnover, portfolio variance, and maximum drawdown.
- **REQ-DRL2.2**: Implement causal episodic training harness with Generalized Advantage Estimation (GAE) and clipped surrogate objective.

#### REQ-DRL3: DRL Benchmark & Verification
- **REQ-DRL3.1**: Provide `examples/demo_dyfo_continuous_drl.py` comparing `DyFO-DRL` vs `Raw-DRL` vs `EWMA-GMVP` vs `1/N`.
- **REQ-DRL3.2**: Implement unit tests in `tests/test_drl_continuous_actor_critic.py`.

---

### Phase 3: Discrete Deep Q-Networks & Dynamic Hedging (Sequence Step 2)

#### REQ-DQN1: Discrete MDP Formulation for Regime Switching
- **REQ-DQN1.1**: Define compact state vector $s_t = [\text{EigVals}(\mathbf{\Sigma}_t), \text{TopCentrality}(\mathbf{Z}_t), \boldsymbol{\pi}_t, \text{Drawdown}_t]$.
- **REQ-DQN1.2**: Define discrete action space $\mathcal{A} = \{a_{\text{Alpha\_GMVP}}, a_{\text{Defensive\_ERC}}, a_{\text{TailRisk\_Hedge}}, a_{\text{SectorRotation}}\}$.

#### REQ-DQN2: Dueling Double-DQN with Prioritized Experience Replay (PER)
- **REQ-DQN2.1**: Implement `DuelingDQNNetwork` separating state value $V(s)$ and action advantage $A(s, a)$.
- **REQ-DQN2.2**: Implement `PrioritizedReplayBuffer` with TD-error sampling ($\delta_i^\alpha$) and importance-sampling weight correction ($w_i = (N \cdot P(i))^{-\beta}$).
- **REQ-DQN2.3**: Implement target network polyak/soft update ($\theta_{\text{target}} \leftarrow \tau \theta + (1-\tau) \theta_{\text{target}}$).

#### REQ-DQN3: DQN Demo & Unit Verification
- **REQ-DQN3.1**: Create `examples/demo_dyfo_dqn_hedging.py` with multi-panel visualization of Q-values, action distribution over market regimes (e.g. COVID shock), and portfolio equity curve.
- **REQ-DQN3.2**: Implement unit tests in `tests/test_dqn_discrete_hedging.py`.

---

## 3. Acceptance & Verification Criteria
1. **Zero Data Leakage**: All state constructors, GraphRAG subgraphs, and reward calculations must be strictly causal ($t \le \text{today}$).
2. **Deterministic Fallbacks**: All LLM and DRL modules must operate with deterministic offline fallbacks (mock LLM / rule-based compiler) to guarantee 100% CI pass rate without external API keys.
3. **Full Pytest Suite**: All unit and integration tests must pass cleanly.
4. **Second Brain & DOC_MASTER Synchronization**: Updated status reflected in Obsidian Vault entities.
