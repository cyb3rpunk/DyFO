# BRACIS Submission Fixes Tasks

**Design**: `.specs/features/bracis_submission_fixes/design.md`
**Status**: Draft

---

## Execution Plan

### Phase 1: Evaluation Integrity (Parallel OK)

Tasks related to improving the statistical and baseline rigor.

```
     ┌→ T1 (Baselines) ────┐
     ├→ T2 (DM Tests) ─────┤
     └→ T3 (ROLAND tune) ──┘
```

### Phase 2: Manuscript Edits (Parallel OK)

Tasks related to fixing the narrative, terminology, and restoring sections.

```
     ┌→ T4 (Terminology) ─────┐
     ├→ T5 (Positioning) ─────┤
     ├→ T6 (Scalability) ─────┤
     └→ T7 (Reproducibility) ─┘
```

---

## Task Breakdown

### T1: Add Persistence & EWMA Baselines [P]

**What**: Implement naive persistence and EWMA baselines, calculate their metrics, and update the evaluation tables.
**Where**: Evaluation scripts and evaluation section of manuscript.
**Depends on**: None
**Reuses**: Existing evaluation pipeline.
**Requirement**: [BRACIS]-01

**Tools**:
- MCP: `filesystem`
- Skill: NONE

**Done when**:
- [ ] Persistence baseline calculated (ρ_{t+1} ≈ ρ_t).
- [ ] EWMA baseline calculated.
- [ ] Evaluation tables in manuscript updated with new baselines.
- [ ] Delta against baselines discussed in text.

**Tests**: none
**Gate**: none

---

### T2: Fix DM Test Clustering [P]

**What**: Recompute Diebold-Mariano tests using day-clustered or per-pair pooled inference.
**Where**: DM test evaluation scripts and manuscript text.
**Depends on**: None
**Reuses**: Existing DM test scripts.
**Requirement**: [BRACIS]-01

**Tools**:
- MCP: `filesystem`
- Skill: NONE

**Done when**:
- [ ] DM test logic updated to avoid cross-sectional dependence inflation.
- [ ] Realistic p-values (not 10^-121) reported in the paper.
- [ ] Cohen's d > 5 claims recomputed and fixed.

**Tests**: none
**Gate**: none

---

### T3: Address ROLAND Baseline [P]

**What**: Tune ROLAND with daily snapshots to fix F1≈0, OR replace it with DySAT/EvolveGCN/TGN.
**Where**: Baseline scripts and manuscript tables.
**Depends on**: None
**Requirement**: [BRACIS]-03

**Tools**:
- MCP: `filesystem`
- Skill: NONE

**Done when**:
- [ ] ROLAND baseline yields a reasonable score OR is replaced entirely.
- [ ] Manuscript tables and text updated accordingly.

**Tests**: none
**Gate**: none

---

### T4: Terminology and Reframing Consistency [P]

**What**: Global search and replace for inconsistent terms ("Operator", "stateless") and remove portfolio-level overclaims.
**Where**: All manuscript files (`.tex` or `.md`).
**Depends on**: None
**Requirement**: [BRACIS]-02

**Tools**:
- MCP: `filesystem`
- Skill: NONE

**Done when**:
- [ ] "Operator" replaced with "Observer" (or vice versa consistently).
- [ ] "stateless" replaced with "non-recurrent" or "BPTT-free, bounded-history attention".
- [ ] "First domain-adapted" / "First rigorous" softened.
- [ ] Abstract reframed around forecasting accuracy rather than portfolio gains.
- [ ] Verified causal cutoff for DCC-GARCH labels is explicitly stated.

**Tests**: none
**Gate**: none

---

### T5: Add Novelty Positioning [P]

**What**: Add 1-2 paragraphs in Related Work distinguishing DyFO from HGT, R-GAT, HAN, RGCN canon.
**Where**: Related Work section of manuscript.
**Depends on**: None
**Requirement**: [BRACIS]-02

**Tools**:
- MCP: `filesystem`
- Skill: NONE

**Done when**:
- [ ] Paragraph clearly distinguishes DyFO's edge-conditioning from standard canonical GNNs.

**Tests**: none
**Gate**: none

---

### T6: Restore N=100 Scalability Table [P]

**What**: Uncomment and restore the scalability section showing N=100 results to preempt the small universe critique.
**Where**: Manuscript scalability/experiments section.
**Depends on**: None
**Requirement**: [BRACIS]-03

**Tools**:
- MCP: `filesystem`
- Skill: NONE

**Done when**:
- [ ] N=100 table and text are restored and render correctly.

**Tests**: none
**Gate**: none

---

### T7: Add Reproducibility Artifacts [P]

**What**: Add anonymized Github repo URL, seed protocol, hyperparameter grid, and runtime per window to the appendix.
**Where**: Manuscript Appendix / Implementation Details.
**Depends on**: None
**Requirement**: [BRACIS]-04

**Tools**:
- MCP: `filesystem`
- Skill: NONE

**Done when**:
- [ ] Anonymous repo URL is added.
- [ ] Seed list, hyperparameter grid, and runtime details are clearly stated.

**Tests**: none
**Gate**: none

---

## Parallel Execution Map

```
Phase 1 & 2 (Parallel):
  ├── T1 [P]
  ├── T2 [P]
  ├── T3 [P]
  ├── T4 [P]
  ├── T5 [P]
  ├── T6 [P]
  └── T7 [P]
```

## Task Granularity Check

| Task                            | Scope         | Status       |
| ------------------------------- | ------------- | ------------ |
| T1: Add Persistence/EWMA        | Eval script + table | ✅ Granular  |
| T2: Fix DM Test                 | Eval script + text  | ✅ Granular  |
| T3: Tune ROLAND                 | Eval script + table | ✅ Granular  |
| T4: Terminology                 | Manuscript text     | ✅ Granular  |
| T5: Novelty Positioning         | Manuscript text     | ✅ Granular  |
| T6: Restore Scalability         | Manuscript text     | ✅ Granular  |
| T7: Add Reproducibility         | Manuscript text     | ✅ Granular  |

## Diagram-Definition Cross-Check

| Task | Depends On (task body) | Diagram Shows | Status |
| ---- | ---------------------- | ------------- | ------ |
| T1 | None | None | ✅ Match |
| T2 | None | None | ✅ Match |
| T3 | None | None | ✅ Match |
| T4 | None | None | ✅ Match |
| T5 | None | None | ✅ Match |
| T6 | None | None | ✅ Match |
| T7 | None | None | ✅ Match |

## Test Co-location Validation

Since this is largely a research evaluation fix + manuscript writing task, code changes will not have standard unit/e2e tests, but rather data validation tests.

| Task | Code Layer Created/Modified | Matrix Requires | Task Says | Status |
| ---- | --------------------------- | --------------- | --------- | ------ |
| T1 | Scripts/Tables              | none            | none      | ✅ OK  |
| T2 | Scripts/Tables              | none            | none      | ✅ OK  |
| T3 | Scripts/Tables              | none            | none      | ✅ OK  |
| T4-7| Manuscript                 | none            | none      | ✅ OK  |
