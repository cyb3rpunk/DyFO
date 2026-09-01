# Task Breakdown: Neuro-Symbolic LLM, Continuous DRL & Discrete DQN

**Feature ID**: `neurosymbolic-drl-dqn-advances`  
**Base Branch**: `feat/neurosymbolic-drl-dqn-advances`  
**Execution Order**: Phase 1 (3: Neuro-Symbolic LLM) $\to$ Phase 2 (1: Continuous DRL) $\to$ Phase 3 (2: Discrete DQN)  

---

## Phase 1: (3) Neuro-Symbolic AI & LLMs (GraphRAG, Triples & Constrained Solver)

- [x] **Task 1.1: Causal Subgraph Extractor & Semantic Triples Serializer**
  - **What**: Build `dyfo/neurosymbolic/subgraph_extractor.py` to extract ego-networks, correlation innovation shocks ($\Delta\hat{\rho}$), macro events, and serialize to JSON-LD, RDF/Turtle, and Human-Readable Triples.
  - **Where**: `dyfo/neurosymbolic/subgraph_extractor.py`
  - **Depends on**: None
  - **Tests**: `tests/test_neurosymbolic_graphrag.py`
  - **Done when**: Subgraph extracted without future look-ahead and serialization formats valid.

- [x] **Task 1.2: GraphRAG Prompt Engine & Mock/Online LLM Reasoner**
  - **What**: Build `dyfo/neurosymbolic/graphrag_prompt_engine.py` with Chain-of-Thought risk attribution prompts and support for OpenAI, Claude, Gemini, and deterministic local mock fallback.
  - **Where**: `dyfo/neurosymbolic/graphrag_prompt_engine.py`
  - **Depends on**: Task 1.1
  - **Tests**: `tests/test_neurosymbolic_graphrag.py`
  - **Done when**: Generates coherent risk explanations and structured JSON symbolic constraints.

- [x] **Task 1.3: Symbolic Constraint Parser & Constrained QP Solver**
  - **What**: Build `dyfo/neurosymbolic/symbolic_parser.py` and `dyfo/neurosymbolic/constrained_solver.py` to compile LLM constraints into linear inequality bounds ($A \mathbf{w} \le \mathbf{b}$) and solve minimum-variance portfolios over DyFO's SPD covariance.
  - **Where**: `dyfo/neurosymbolic/symbolic_parser.py`, `dyfo/neurosymbolic/constrained_solver.py`
  - **Depends on**: Task 1.2
  - **Tests**: `tests/test_neurosymbolic_graphrag.py`
  - **Done when**: Quadratic program solves strictly within bounds and guarantees $\mathbf{w} \ge 0, \sum w_i = 1$.

- [x] **Task 1.4: Phase 1 End-to-End Demo & Test Suite**
  - **What**: Build `examples/demo_dyfo_llm_neurosymbolic.py` executing 1-year walk-forward evaluation, saving reports and figures.
  - **Where**: `examples/demo_dyfo_llm_neurosymbolic.py`, `tests/test_neurosymbolic_graphrag.py`
  - **Depends on**: Task 1.3
  - **Done when**: Demo runs with exit code 0 and all unit tests pass.

---

## Phase 2: (1) Continuous Deep Reinforcement Learning (Actor-Critic / PPO)

- [x] **Task 2.1: Continuous Relational State Constructor**
  - **What**: Build `dyfo/drl/continuous_state.py` packaging dynamic graph embeddings $\mathbf{Z}_t \in \mathbb{R}^{N \times 100}$, current portfolio weights $\mathbf{w}_{t-1}$, and macro regime probabilities $\boldsymbol{\pi}_t$.
  - **Where**: `dyfo/drl/continuous_state.py`
  - **Depends on**: None
  - **Tests**: `tests/test_drl_continuous_actor_critic.py`
  - **Done when**: State tensor constructed with correct dimensions and zero look-ahead bias.

- [x] **Task 2.2: Relational Cross-Attention Actor-Critic Policy Network**
  - **What**: Build `dyfo/drl/relational_actor_critic.py` implementing multi-head attention over asset representations to output simplex weights $\mathbf{w}_t \in \Delta^N$ and baseline value $V(s)$.
  - **Where**: `dyfo/drl/relational_actor_critic.py`
  - **Depends on**: Task 2.1
  - **Tests**: `tests/test_drl_continuous_actor_critic.py`
  - **Done when**: Gradient propagates to both policy and value heads without NaN.

- [x] **Task 2.3: Risk-Regularized PPO Trainer & Continuous Demo**
  - **What**: Build `dyfo/drl/ppo_trainer.py` and `examples/demo_dyfo_continuous_drl.py` with multi-objective penalties (turnover, variance, drawdown).
  - **Where**: `dyfo/drl/ppo_trainer.py`, `examples/demo_dyfo_continuous_drl.py`
  - **Depends on**: Task 2.2
  - **Tests**: `tests/test_drl_continuous_actor_critic.py`
  - **Done when**: DyFO-DRL trains stably and outperforms Raw-DRL in Sharpe and turnover.

---

## Phase 3: (2) Discrete Deep Q-Networks (DQN & Dynamic Hedging)

- [x] **Task 3.1: Discrete MDP State Constructor & Action Space**
  - **What**: Build `dyfo/dqn/discrete_state.py` creating compact topological state vectors $s_t$ and defining 4 discrete macro/hedging actions.
  - **Where**: `dyfo/dqn/discrete_state.py`
  - **Depends on**: None
  - **Tests**: `tests/test_dqn_discrete_hedging.py`
  - **Done when**: State space maps eigen-spectrum and graph centrality cleanly.

- [x] **Task 3.2: Dueling Double-DQN Network & Prioritized Replay Buffer**
  - **What**: Build `dyfo/dqn/dueling_dqn.py` and `dyfo/dqn/prioritized_replay.py` with value/advantage separation and TD-error proportional sampling.
  - **Where**: `dyfo/dqn/dueling_dqn.py`, `dyfo/dqn/prioritized_replay.py`
  - **Depends on**: Task 3.1
  - **Tests**: `tests/test_dqn_discrete_hedging.py`
  - **Done when**: Replay buffer samples transitions with correct importance-sampling weights.

- [x] **Task 3.3: DQN Agent, Dynamic Hedging Demo & Verification Suite**
  - **What**: Build `dyfo/dqn/dqn_agent.py` and `examples/demo_dyfo_dqn_hedging.py` simulating dynamic regime switching over historical stress regimes (e.g. COVID-19).
  - **Where**: `dyfo/dqn/dqn_agent.py`, `examples/demo_dyfo_dqn_hedging.py`, `tests/test_dqn_discrete_hedging.py`
  - **Depends on**: Task 3.2
  - **Done when**: DQN agent learns non-trivial regime switching policy, demo runs with exit code 0, and unit tests pass.

---

## Phase 4: Integration, SOTA Validation & Second Brain Sync

- [x] **Task 4.1: Full Repository Regression Test Suite**
  - **What**: Run complete `pytest` suite ensuring all 73 tests pass with 100% green status.
- [x] **Task 4.2: Second Brain Vault & DOC_MASTER Synchronization**
  - **What**: Update `entities/DyFO.md`, `Pending Tasks.md`, and DOC_MASTER status logs.
