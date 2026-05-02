# Delta-Rho Target (Δρ) Tasks

**Design**: `.specs/features/delta_rho_target/design.md`
**Status**: Draft

---

## Execution Plan

```
Phase 1: Core Data Layer (sequential — T2 depends on T1)
  T1: build_delta_regression_labels + CorrelationRegressor delta mode
  T2: compute_regression_metrics reconstruction extension

Phase 2: Config & Baselines (parallel — all independent of each other)
  T3: DyFOConfig.use_delta_target flag + VALID_MODEL_VARIANTS
  T4: zero baseline
  T5: delta_ewma baseline

Phase 3: Integration (sequential — depends on Phase 1 & 2)
  T6: Thread use_delta_target through run_split training loop
  T7: CLI flag --delta_target in eval script + sanity assertion

Phase 4: Manuscript (independent)
  T8: Write Δρ reformulation section in samplepaper.tex
```

---

## Task Breakdown

### T1: `build_delta_regression_labels` + decoder delta mode [SEQUENTIAL FIRST]

**What**: Add `build_delta_regression_labels(corr_tomorrow, corr_today, num_nodes, sample_ratio)`
to `dyfo/core/link_prediction.py`. Add `output_mode: str = "absolute"` param to
`CorrelationRegressor.__init__` — when `"delta"`, remove the `tanh` from the output layer
(replace with `nn.Identity()` or simply omit the activation in `self.net`).

**Where**: `dyfo/core/link_prediction.py`

**Depends on**: None

**Requirement**: [DELTA]-01

**Implementation detail**:
- Only pairs present in BOTH `corr_tomorrow` and `corr_today` are included.
- `delta = corr_tomorrow[(i,j)] - corr_today[(i,j)]` for each qualifying pair.
- `CorrelationRegressor` with `output_mode="delta"`: final layer has no activation.
  Modify `self.net` to drop the `nn.Tanh()` at the end.

**Done when**:
- [ ] `build_delta_regression_labels` is importable from `dyfo.core.link_prediction`.
- [ ] Given identical `corr_today` and `corr_tomorrow`, all returned delta values are 0.0.
- [ ] `CorrelationRegressor(output_mode="delta")` produces predictions outside [-1, 1] for
  large inputs (i.e., tanh is not applied).
- [ ] `CorrelationRegressor(output_mode="absolute")` behaves identically to current code.

---

### T2: `compute_regression_metrics` reconstruction extension [DEPENDS ON T1]

**What**: Add optional `rho_today: Optional[torch.Tensor] = None` param.
When provided, compute `rho_reconstructed = rho_today + predictions` and append
`r_squared_reconstructed` and `mae_reconstructed` to the returned dict.

**Where**: `dyfo/core/link_prediction.py:compute_regression_metrics`

**Depends on**: T1 (needs the delta concept to be established)

**Requirement**: [DELTA]-03

**Implementation detail**:
- The `rho_today` tensor must be aligned with `predictions` and `targets` (same length,
  same pair order) — callers are responsible for alignment.
- When `rho_today is None` (default), the function returns the current dict unchanged.

**Done when**:
- [ ] `compute_regression_metrics(preds, targets, rho_today=rho_t)` returns a dict
  containing `r_squared_reconstructed` and `mae_reconstructed`.
- [ ] **Sanity check**: for `preds = torch.zeros_like(targets)` (zero baseline) with
  `rho_today` = corr_today values, `mae_reconstructed` equals `persistence.mae` from
  the existing run (within 1e-4 tolerance).

---

### T3: Config flag + valid variants [PARALLEL]

**What**: Add `use_delta_target: bool = False` to `DyFOConfig` dataclass in `dyfo/config.py`.
Add `"zero"` and `"delta_ewma"` to `VALID_MODEL_VARIANTS`.

**Where**: `dyfo/config.py`

**Depends on**: None

**Requirement**: [DELTA]-01, [DELTA]-02

**Done when**:
- [ ] `DyFOConfig(use_delta_target=True)` does not raise.
- [ ] `DyFOConfig(model_variant="zero")` does not raise.
- [ ] `DyFOConfig(model_variant="delta_ewma")` does not raise.
- [ ] `DyFOConfig()` (defaults) behaves identically to current code.

---

### T4: `zero` baseline [PARALLEL]

**What**: Implement `zero` baseline in the `is_baseline` branch of `run_split` in
`scripts/train_link_prediction.py`. Predict `Δρ = 0` for all pairs.

**Where**: `scripts/train_link_prediction.py` — `is_baseline` branch alongside `persistence`/`ewma`

**Depends on**: None (but will only produce meaningful results when T6 is done)

**Requirement**: [DELTA]-02

**Implementation detail**:
```python
elif model_variant == "zero":
    preds = torch.zeros(len(src), dtype=torch.float32, device=device)
```

The `zero` baseline is only semantically valid when `use_delta_target=True`.
Add a warning log if `use_delta_target=False` and `model_variant="zero"`.

**Done when**:
- [ ] `--variants zero --delta_target` runs without error.
- [ ] All predicted values in the output are exactly 0.0.
- [ ] `mae_reconstructed` for `zero` matches `persistence.mae` (see T2 sanity check).

---

### T5: `delta_ewma` baseline [PARALLEL]

**What**: Implement `delta_ewma` baseline. Maintains per-pair EWMA state over
historical Δρ values. Prediction for pair (i,j) at time t is the EWMA of past deltas.

**Where**: `scripts/train_link_prediction.py` — `is_baseline` branch

**Depends on**: None (but requires T6 for meaningful integration)

**Requirement**: [DELTA]-02

**Implementation detail**:
- Requires tracking the previous day's `corr_today` to compute `delta_today = rho_t - rho_{t-1}`.
- State: `delta_ewma_state: Dict[(i,j), float] = {}`, initialized to 0.0.
- Update: `delta_ewma_state[(i,j)] = alpha * delta_today + (1-alpha) * delta_ewma_state.get((i,j), 0.0)`.
- Use `ewma_alpha` from config (same as the existing `ewma` baseline).
- The loop must store `prev_corr_today` from the previous iteration.

**Done when**:
- [ ] `--variants delta_ewma --delta_target` runs without error.
- [ ] EWMA state updates across days (verify state is non-zero after day 2).
- [ ] `delta_ewma` R² on Δρ series is higher than `zero` if any momentum exists.

---

### T6: Thread `use_delta_target` through training loop [DEPENDS ON T1, T2, T3]

**What**: Modify `run_split` in `scripts/train_link_prediction.py` to branch on
`use_delta_target`. When True:
1. Call `build_delta_regression_labels(corr_tomorrow, corr_today, num_nodes)`.
2. Build `rho_today_tensor` aligned with `src`/`dst` for reconstruction.
3. Instantiate `CorrelationRegressor(output_mode="delta")`.
4. Pass `rho_today=rho_today_tensor` to `compute_regression_metrics`.

**Where**: `scripts/train_link_prediction.py`

**Depends on**: T1, T2, T3

**Requirement**: [DELTA]-01, [DELTA]-03

**Implementation detail**:
- `corr_today` is already accessed at line ~617 for `ewma`/`persistence` baselines.
  Lift this lookup to be unconditional when `use_delta_target=True`.
- `rho_today_tensor`: for each (src[k], dst[k]) pair, look up `corr_today.get((s,d), 0.0)`.
  This tensor is passed to `compute_regression_metrics` for reconstruction.

**Done when**:
- [ ] `train_link_prediction(..., use_delta_target=True)` runs end-to-end for `tgn` variant.
- [ ] Returned metrics dict contains `r_squared` (on Δρ) AND `r_squared_reconstructed` (on ρ).
- [ ] Running with `use_delta_target=False` produces bit-identical results to current baseline.

---

### T7: CLI flag + eval script integration + sanity assertion [DEPENDS ON T3, T6]

**What**: Add `--delta_target` CLI flag to `run_bootstrap_eval_temporal_kg_rev3.py`.
Propagate to config and `_train_window`. Add sanity assertion: after running `zero` and
`persistence` variants, assert `zero.mae_reconstructed ≈ persistence.mae`.

**Where**: `scripts/run_bootstrap_eval_temporal_kg_rev3.py`

**Depends on**: T3, T6

**Requirement**: [DELTA]-01, [DELTA]-02

**Done when**:
- [ ] `python scripts/run_bootstrap_eval_temporal_kg_rev3.py --variants zero delta_ewma tgn --delta_target --n_tickers 50`
  completes without error.
- [ ] Summary JSON contains `r_squared` (Δρ) and `r_squared_reconstructed` (ρ) for all variants.
- [ ] Sanity assertion fires if `zero.mae_reconstructed` deviates > 1e-4 from `persistence.mae`.

---

### T8: Manuscript — Δρ reformulation section [PARALLEL, independent]

**What**: Add a subsection in `doc/samplepaper.tex` within the Experiments section
explaining the Δρ reformulation.

**Where**: `doc/samplepaper.tex`

**Depends on**: None

**Requirement**: [DELTA]-04

**Content outline**:
1. Motivation: DCC-GARCH labels are highly autocorrelated (ρ ≈ 0.90–0.95 day-to-day).
   Statistical baselines with direct ρ_t access trivially achieve high absolute R².
2. Reformulation: target = Δρ_{t+1} = ρ_{t+1} − ρ_t. Trivial baseline becomes "predict 0".
3. Information symmetry: all models (DyFO and baselines) start from the same mean-reversion prior.
4. Reconstruction: ρ̂_{t+1} = ρ_t + Δρ̂_{t+1}; report both Δρ R² and reconstructed ρ R².
5. Scientific claim: DyFO's graph structure and event stream capture structural shocks that
   the mean-reversion prior cannot; this is where the contribution lies.

**Done when**:
- [ ] Subsection compiles in LaTeX without errors.
- [ ] Section clearly distinguishes DyFO-ρ (absolute) from DyFO-Δρ experiments.
- [ ] Information asymmetry argument is stated explicitly.

---

## Parallel Execution Map

```
Phase 1 (sequential):
  T1 → T2

Phase 2 (parallel, start after T1):
  T3, T4, T5 (all independent)

Phase 3 (sequential, start after Phase 1 & 2):
  T6 → T7

Phase 4 (parallel, any time):
  T8
```

```
T1 ──→ T2 ──┐
             ├──→ T6 ──→ T7
T3 ──────────┤
T4 ──────────┤
T5 ──────────┘

T8 (independent)
```

---

## Task Granularity Check

| Task | Scope | Status |
|------|-------|--------|
| T1: Label builder + decoder mode | `link_prediction.py` — 2 functions | ✅ Granular |
| T2: Reconstruction metrics | `link_prediction.py` — 1 function extension | ✅ Granular |
| T3: Config flag | `config.py` — 2 lines | ✅ Granular |
| T4: `zero` baseline | `train_link_prediction.py` — 3 lines | ✅ Granular |
| T5: `delta_ewma` baseline | `train_link_prediction.py` — ~15 lines | ✅ Granular |
| T6: Training loop threading | `train_link_prediction.py` — conditional branches | ✅ Granular |
| T7: CLI + sanity assertion | `run_bootstrap_eval_temporal_kg_rev3.py` | ✅ Granular |
| T8: Manuscript section | `samplepaper.tex` — 1 subsection | ✅ Granular |

---

## Diagram-Definition Cross-Check

| Task | Depends On (body) | Diagram Shows | Status |
|------|-------------------|---------------|--------|
| T1 | None | None | ✅ Match |
| T2 | T1 | T1 → T2 | ✅ Match |
| T3 | None | None | ✅ Match |
| T4 | None | None | ✅ Match |
| T5 | None | None | ✅ Match |
| T6 | T1, T2, T3 | Phase 1&2 → T6 | ✅ Match |
| T7 | T3, T6 | T6 → T7 | ✅ Match |
| T8 | None | Independent | ✅ Match |
