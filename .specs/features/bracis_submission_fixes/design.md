# BRACIS Submission Fixes Design

**Spec**: `.specs/features/bracis_submission_fixes/spec.md`
**Status**: Draft

---

## Architecture Overview

This feature encompasses targeted improvements across the manuscript and evaluation scripts to satisfy BRACIS reviewer expectations. The design involves analytical additions (baselines, DM tests) and textual revisions (positioning, terminology).

## Code Reuse Analysis

### Existing Components to Leverage

| Component            | Location            | How to Use                |
| -------------------- | ------------------- | ------------------------- |
| Evaluation Scripts   | `src/evaluation/` (or similar) | Extend to calculate naive Persistence (ρ_{t+1} ≈ ρ_t) and EWMA |
| DM Test Script       | `src/evaluation/`   | Update the statistical inference logic to pool per day or block-bootstrap |
| Scalability Section  | `paper.tex`/`paper.md` | Uncomment the existing N=100 table and discussion |

---

## Components

### 1. Baseline Evaluator
- **Purpose**: Calculate Persistence and EWMA baselines to ground the R²=0.893 result.
- **Location**: Evaluation codebase.
- **Interfaces**:
  - `calculate_persistence(labels)`
  - `calculate_ewma(series, span)`
- **Dependencies**: Target labels (DCC-GARCH series).

### 2. Statistical Tester (DM Test)
- **Purpose**: Implement cluster-robust or pair-blocked bootstrap inference for the Diebold-Mariano test.
- **Location**: Evaluation codebase.
- **Interfaces**:
  - `clustered_dm_test(preds1, preds2, labels, cluster_idx)`
- **Dependencies**: Predictions from models, target labels.

### 3. Manuscript Revisions (Textual)
- **Purpose**: Address terminology, novelty positioning, reproducibility, and remove overclaims.
- **Location**: LaTeX/Markdown manuscript files.
- **Modifications**:
  - Search/Replace "Operator" -> "Observer" (or vice versa).
  - Search/Replace "stateless" -> "non-recurrent".
  - Related Work paragraph: Contrast DyFO with HGT, R-GAT, HAN, RGCN.
  - Appendix: Add anonymous Github link, seed list, hyperparams, compute budget.

### 4. ROLAND Tuning / Replacement
- **Purpose**: Fix the suspicious F1=0 baseline.
- **Location**: Model training scripts / baselines.
- **Modifications**: Either adjust ROLAND hyperparameters to use daily snapshots correctly or swap to DySAT/EvolveGCN/TGN.

---

## Error Handling Strategy

| Error Scenario | Handling      | User Impact      |
| -------------- | ------------- | ---------------- |
| Label leakage detected in DCC-GARCH | Explicitly state DCC fitting cutoff per window and prove event timestamp t < label timestamp t+1 in the text. | Reassures reviewers about causality. |

---

## Tech Decisions

| Decision          | Choice          | Rationale     |
| ----------------- | --------------- | ------------- |
| ROLAND strategy | To be determined during task execution | Tuning ROLAND is faster if it's just a hyperparam issue; replacing is better if the architecture fundamentally mismatches the data. |
| Terminology | Standardize on "Observer" (or whatever user prefers) | Must pick one and be perfectly consistent. |
| Portfolio framing | Reframe as pure correlation forecasting | Avoids the need to implement a differentiable GMV head before the deadline. |
