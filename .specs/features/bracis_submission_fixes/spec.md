# BRACIS Submission Fixes Specification

## Problem Statement

The current paper (DyFO) has an acceptance likelihood of ~30-40% for BRACIS due to missing baselines, inconsistent terminology, missing novelty positioning, and lack of reproducible artifacts. These issues must be fixed to raise the acceptance rate to ~60% and avoid the top 3 rejection risks.

## Goals

- [ ] Add Persistence and EWMA baselines and report delta against them.
- [ ] Ensure consistent terminology (e.g., DyFO = "Observer") and fix the incorrect "stateless" claim.
- [ ] Position the paper correctly vs. HGT/HAN/R-GAT/RGCN in the Related Work.
- [ ] Restore the scalability table for N=100 to address "small universe" critique.
- [ ] Address the ROLAND baseline (tune it or replace it).
- [ ] Provide an anonymized code repository link and hyperparameter details.
- [ ] Recompute DM tests with proper day-clustered inference.

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
| ----------- | -------------- |
| Differentiable GMV head | Too large a scope change pre-deadline. Better to reframe the paper as pure correlation forecasting. |
| Expand to N=500 | DCC-GARCH cost is real. N=100 with existing window count is sufficient. |
| Re-running everything with more seeds | Fix methodology gaps first; seed variance is secondary if persistence baseline collapses the gap. |
| Polishing prose extensively | Substance is the bottleneck; fixing major flaws is higher ROI than prose. |

---

## User Stories

### P1: Add Baselines and Significance ⭐ MVP

**User Story**: As a BRACIS reviewer, I want to see naive persistence and EWMA baselines so that I can evaluate true model learning over simple heuristics.

**Why P1**: Fatal issue. Without it, R²=0.893 is uninterpretable and collapses credibility.

**Acceptance Criteria**:

1. WHEN evaluation tables are presented THEN Persistence baseline metrics SHALL be included.
2. WHEN evaluation tables are presented THEN EWMA baseline metrics SHALL be included.
3. WHEN DM test is reported THEN it SHALL use day-clustered or per-pair pooled inference to avoid implausible p-values.

**Independent Test**: Can view the evaluation table and verify Persistence, EWMA, and realistic DM p-values.

---

### P1: Positioning and Terminology ⭐ MVP

**User Story**: As a reader, I want consistent terminology and clear novelty positioning so that I can understand the exact contribution.

**Why P1**: Inconsistencies signal an unfinished paper; missing positioning vs Relation-aware GNNs makes it seem like simple engineering.

**Acceptance Criteria**:

1. WHEN reading the paper THEN DyFO SHALL be consistently referred to as "Observer" (or the chosen term).
2. WHEN reading the paper THEN "stateless" SHALL be replaced with "non-recurrent" or "BPTT-free, bounded-history attention".
3. WHEN reading Related Work THEN there SHALL be a paragraph distinguishing DyFO's edge-conditioning from HGT/HAN/R-GAT/RGCN canon.
4. WHEN reading abstract/intro THEN overclaiming statements ("first domain-adapted", "first rigorous") SHALL be softened or removed.

**Independent Test**: Text search yields zero instances of inconsistent terminology or overclaiming statements.

---

### P2: Evaluation Integrity

**User Story**: As an evaluator, I want proper baselines and scalability proofs to trust the system's robustness.

**Why P2**: ROLAND F1 ≈ 0 in 7/9 windows is suspicious; N=50 universe is smaller than standard portfolio papers.

**Acceptance Criteria**:

1. WHEN viewing baseline results THEN ROLAND SHALL be tuned properly OR replaced with a stronger baseline (DySAT/EvolveGCN/TGN).
2. WHEN reading the scalability section THEN the N=100 table SHALL be restored and visible.

**Independent Test**: Verify the presence of the N=100 table and valid ROLAND (or replacement) metrics.

---

### P3: Reproducibility Artifacts

**User Story**: As a reproducibility chair, I want code and hyperparameter details to verify the results.

**Why P3**: BRACIS ML track reviewers increasingly enforce reproducibility standards.

**Acceptance Criteria**:

1. WHEN reading the paper THEN an anonymous repo URL SHALL be provided.
2. WHEN reviewing the appendix THEN explicit seed lists, hyperparameter grids, and runtime per window SHALL be included.

---

## Requirement Traceability

| Requirement ID | Story       | Phase  | Status  |
| -------------- | ----------- | ------ | ------- |
| [BRACIS]-01    | P1: Baselines & DM | Design | Pending |
| [BRACIS]-02    | P1: Terminology & Positioning | Design | Pending |
| [BRACIS]-03    | P2: Evaluation Integrity | Design | Pending |
| [BRACIS]-04    | P3: Reproducibility | Design | Pending |

---

## Success Criteria

How we know the feature is successful:

- [ ] Persistence and EWMA baselines are fully calculated and included in results.
- [ ] DM tests output realistic p-values.
- [ ] N=100 scalability section is restored.
- [ ] Inconsistent terms ("Operator", "stateless") are completely eradicated from the text.
