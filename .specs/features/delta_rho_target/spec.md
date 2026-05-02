# Delta-Rho Target (Δρ) Specification

**Feature**: `delta_rho_target`
**Status**: Draft
**Related**: `.specs/features/bracis_submission_fixes/spec.md`

---

## Problem Statement

DyFO currently predicts ρ_{t+1} directly (absolute correlation). As confirmed empirically,
this puts DyFO at a structural disadvantage against statistical baselines (Persistence, EWMA)
that have direct oracle access to ρ_t as a scalar lookup. The competition is asymmetric:
baselines copy a highly autocorrelated scalar; DyFO reconstructs it from graph signals.

The Δρ approach reformulates the target as:

```
Δρ_{t+1} = ρ_{t+1} - ρ_t
```

Under this formulation:
- The trivial naive baseline is "predict 0" (no change) — a mean-reversion prior.
- DyFO must predict structural shocks (the non-trivial part of correlation dynamics).
- Persistence and EWMA become fair competitors: they also predict the delta.
- The information asymmetry argument is resolved: all models start from the same prior.

Reconstructed prediction at inference time: ρ̂_{t+1} = ρ_t + Δρ̂_{t+1}, so interpretability
in the original ρ space is preserved.

---

## Goals

- [ ] Extend label construction to support Δρ = ρ_{t+1} − ρ_t targets.
- [ ] Add `use_delta_target` config flag to DyFOConfig (non-breaking, default False).
- [ ] Add `zero` and `delta_ewma` baselines that compete fairly on the Δρ problem.
- [ ] Adapt the decoder output activation for the Δρ range (linear head, no tanh).
- [ ] Extend `compute_regression_metrics` to report both Δρ metrics and reconstructed ρ metrics.
- [ ] Add a `delta_rho_target` variant to the eval scripts so experiments can be run end-to-end.
- [ ] Document the architectural choice and scientific rationale in the manuscript.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Differentiable GMV / portfolio head | Separate concern; Δρ solves the fair-comparison problem, not portfolio utility. |
| Multi-step Δρ forecasting (t+2, t+k) | Out of scope for current paper; single-step Δρ is the contribution. |
| Change to encoder architecture | Encoder stays the same; only target and decoder output change. |
| Retraining all existing models with Δρ | Only DyFO (tgn, tgat) needs Δρ variants; ROLAND/GAT-static are not the focus. |

---

## User Stories

### P1: Δρ Target Construction ⭐ MVP

**User Story**: As a researcher, I want DyFO to be trained against Δρ_{t+1} = ρ_{t+1} − ρ_t
so that the model must learn structural correlation shocks rather than the autocorrelation baseline.

**Why P1**: Without this, the comparison against Persistence/EWMA is asymmetric and the
scientific claim that DyFO learns beyond persistence is not testable.

**Acceptance Criteria**:

1. WHEN `use_delta_target=True` THEN `build_delta_regression_labels` SHALL return
   `(src, dst, rho_{t+1} - rho_t)` for all known pairs present in both today and tomorrow.
2. WHEN `use_delta_target=False` THEN behaviour SHALL be identical to current codebase.
3. WHEN a pair exists in tomorrow but not in today THEN it SHALL be skipped (no imputation).
4. WHEN `use_delta_target=True` THEN the decoder output layer SHALL use linear activation
   (not tanh) to allow unbounded Δρ predictions.

**Independent Test**: Run `python scripts/run_bootstrap_eval_temporal_kg_rev3.py --variants tgn --delta_target`
and verify that `mean_window_metrics.tgn.r_squared` is computed on the Δρ series.

---

### P1: Fair Baselines for Δρ ⭐ MVP

**User Story**: As a BRACIS reviewer, I want to see Δρ-appropriate baselines so that I can
judge whether DyFO learns anything beyond naive mean-reversion.

**Why P1**: The Δρ problem's trivial baseline is "predict 0", not "copy ρ_t". Without this,
the new evaluation has the same gap as before.

**Acceptance Criteria**:

1. WHEN `variant=zero` THEN the baseline SHALL predict Δρ = 0 for all pairs (mean-reversion prior).
2. WHEN `variant=delta_ewma` THEN the baseline SHALL maintain an EWMA over past Δρ values
   per pair, predicting the smoothed historical delta.
3. WHEN evaluation tables are presented THEN both baselines SHALL appear alongside DyFO-Δρ results.

**Independent Test**: `zero` baseline R² on Δρ series should be near 0 (predicting the mean of a
zero-mean process). `delta_ewma` should have slightly positive R² if momentum exists.

---

### P2: Reconstructed-ρ Metrics

**User Story**: As a reviewer, I want to see DyFO's performance back in the original ρ space
(after ρ̂_{t+1} = ρ_t + Δρ̂_{t+1} reconstruction) so that results are comparable to prior work.

**Why P2**: Papers in the literature report R² on absolute ρ. Reporting only Δρ metrics
makes it impossible to compare to prior results.

**Acceptance Criteria**:

1. WHEN `use_delta_target=True` THEN `compute_regression_metrics` SHALL also return
   `r_squared_reconstructed` and `mae_reconstructed` on the ρ_{t+1} scale.
2. WHEN the evaluation summary JSON is written THEN BOTH the Δρ metrics AND the
   reconstructed ρ metrics SHALL be included.

**Independent Test**: `mae_reconstructed` for the `zero` baseline should equal Persistence's MAE
(since predicting Δρ=0 is equivalent to predicting ρ̂_{t+1} = ρ_t).

---

### P3: Manuscript Integration

**User Story**: As an author, I want the paper to clearly explain the Δρ reformulation and
its scientific rationale so that reviewers understand why this is a stronger experimental design.

**Why P3**: Without the narrative, reviewers may see the target change as suspicious reframing.

**Acceptance Criteria**:

1. WHEN reading the experiments section THEN there SHALL be a subsection explaining
   Δρ reformulation, the information asymmetry argument, and what the `zero` baseline represents.
2. WHEN reading the results table THEN DyFO-Δρ SHALL be clearly identified as a different
   experimental condition from DyFO-ρ (both reported for transparency).

---

## Requirement Traceability

| Requirement ID  | Story                      | Phase  | Status  |
|-----------------|----------------------------|--------|---------|
| [DELTA]-01      | P1: Δρ Target Construction | Design | Pending |
| [DELTA]-02      | P1: Fair Baselines for Δρ  | Design | Pending |
| [DELTA]-03      | P2: Reconstructed-ρ Metrics| Design | Pending |
| [DELTA]-04      | P3: Manuscript Integration | Design | Pending |

---

## Success Criteria

- [ ] `build_delta_regression_labels` exists and is tested.
- [ ] `use_delta_target=True` runs end-to-end without errors.
- [ ] `zero` and `delta_ewma` baselines produce valid metrics.
- [ ] `r_squared_reconstructed` in summary JSON matches expected value for `zero` baseline.
- [ ] Manuscript section explaining Δρ approach is written.
