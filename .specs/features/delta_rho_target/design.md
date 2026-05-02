# Delta-Rho Target (Δρ) Design

**Spec**: `.specs/features/delta_rho_target/spec.md`
**Status**: Draft

---

## Architecture Overview

The Δρ approach is a **target engineering change**, not an architectural change.
The encoder (TGAT/TGN) is unchanged. Only the label construction, decoder output
activation, and evaluation metrics need modification.

```
Current flow:
  corr_labels_by_date[tomorrow] → build_regression_labels → target: ρ_{t+1}
  decoder: tanh output ∈ [-1, 1]

Proposed flow:
  corr_labels_by_date[today] + corr_labels_by_date[tomorrow]
    → build_delta_regression_labels → target: Δρ = ρ_{t+1} − ρ_t
  decoder: linear output ∈ (-∞, +∞), range in practice ≈ [-0.3, 0.3]
  inference: ρ̂_{t+1} = ρ_t + Δρ̂_{t+1}
```

---

## Code Reuse Analysis

### Existing Components to Modify

| Component | Location | Change |
|-----------|----------|--------|
| `build_regression_labels` | `dyfo/core/link_prediction.py:251` | Add sibling function `build_delta_regression_labels` |
| `CorrelationRegressor` | `dyfo/core/link_prediction.py:217` | Add `output_mode` param: `"absolute"` (tanh) vs `"delta"` (linear) |
| `compute_regression_metrics` | `dyfo/core/link_prediction.py:311` | Extend to accept optional `rho_today` for reconstruction metrics |
| `DyFOConfig` | `dyfo/config.py` | Add `use_delta_target: bool = False` |
| `VALID_MODEL_VARIANTS` | `dyfo/config.py` | Add `"zero"` and `"delta_ewma"` |
| `train_link_prediction` | `scripts/train_link_prediction.py` | Thread `use_delta_target` through `run_split`; pass `corr_today` to label builder |
| `run_bootstrap_eval_temporal_kg_rev3.py` | `scripts/` | Add `--delta_target` CLI flag; add `zero`/`delta_ewma` to `ALL_VARIANTS` |

---

## Components

### 1. `build_delta_regression_labels`

**Location**: `dyfo/core/link_prediction.py`

**Purpose**: Build training targets as Δρ = ρ_{t+1} − ρ_t for all pairs
present in both `corr_today` and `corr_tomorrow`.

**Signature**:
```python
def build_delta_regression_labels(
    corr_tomorrow: dict,   # {(i,j): rho_{t+1}}
    corr_today: dict,      # {(i,j): rho_t}    — needed for delta
    num_nodes: int,
    sample_ratio: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Returns: src, dst, delta_rho (= rho_{t+1} - rho_t)
```

**Invariant**: Only pairs present in BOTH dicts are included.
Pairs only in `corr_tomorrow` (new edges) are excluded — their delta is undefined.

---

### 2. `CorrelationRegressor` — output_mode param

**Location**: `dyfo/core/link_prediction.py:217`

**Change**: Add `output_mode: str = "absolute"` constructor param.
- `"absolute"`: current behaviour (`tanh` output, range [-1, 1]).
- `"delta"`: linear output (no activation). Range is unconstrained; in practice
  ‖Δρ‖ < 0.3 for daily financial correlations.

Rationale for linear output on delta: tanh would artificially clip small shocks.
The Huber loss already provides robustness to outliers; no activation needed.

---

### 3. `compute_regression_metrics` — reconstruction extension

**Location**: `dyfo/core/link_prediction.py:311`

**Change**: Add optional `rho_today: Optional[torch.Tensor] = None` param.
When provided:
```python
rho_reconstructed = rho_today + predictions   # ρ̂_{t+1} = ρ_t + Δρ̂
# Compute r_squared_reconstructed, mae_reconstructed against rho_tomorrow
```

Returned dict gains two optional keys: `r_squared_reconstructed`, `mae_reconstructed`.
Existing callers (absolute mode) pass `rho_today=None` and see no change.

---

### 4. `DyFOConfig` — new flag

**Location**: `dyfo/config.py`

```python
use_delta_target: bool = False
```

Validated in `__post_init__`: no constraint beyond type.

---

### 5. New baselines: `zero` and `delta_ewma`

**Location**: `scripts/train_link_prediction.py` (alongside existing `persistence`/`ewma`)

| Variant | Logic | What it tests |
|---------|-------|---------------|
| `zero` | Predict Δρ = 0 for all pairs | Mean-reversion prior; DyFO-Δρ must beat this |
| `delta_ewma` | EWMA over historical Δρ per pair | Momentum signal in correlation changes |

Both operate in the `is_baseline` branch; no encoder or decoder used.

**`zero` baseline**: `preds = torch.zeros_like(targets)` — trivially simple, but it is
the correct naive baseline for the Δρ problem (equivalent to predicting ρ̂_{t+1} = ρ_t).

**`delta_ewma` baseline**: Maintains `delta_ewma_state: dict[(i,j) → float]`.
On each day: `delta_today = rho_today[(i,j)] - rho_yesterday[(i,j)]`; then
`delta_ewma_state[(i,j)] = alpha * delta_today + (1 - alpha) * delta_ewma_state[(i,j)]`.

Note: `delta_ewma` requires access to two consecutive `corr_today` dicts, so the
baseline's `run_split` loop must track the previous day's correlation.

---

### 6. Training loop change in `run_split`

**Location**: `scripts/train_link_prediction.py:run_split`

When `use_delta_target=True`:
- Replace `build_regression_labels(corr_tomorrow, ...)` with
  `build_delta_regression_labels(corr_tomorrow, corr_today, ...)`.
- Pass `corr_today_tensor` (rho values at positions matching src/dst) to
  `compute_regression_metrics` for reconstruction.
- Decoder instantiated with `output_mode="delta"`.

The `corr_today` lookup is already available in the loop at line ~617 (used by the
existing `ewma` baseline). No new data access required.

---

### 7. CLI flag `--delta_target`

**Location**: `scripts/run_bootstrap_eval_temporal_kg_rev3.py`

Add `parser.add_argument("--delta_target", action="store_true", default=False)`.
Propagate to `DyFOConfig(use_delta_target=args.delta_target)` and pass through
`_train_window` → `train_link_prediction`.

---

## Error Handling Strategy

| Scenario | Handling |
|----------|----------|
| Pair in tomorrow but not today | Skip silently in `build_delta_regression_labels` (delta undefined) |
| `rho_today=None` passed to `compute_regression_metrics` with delta mode | Log warning; skip reconstruction metrics (don't crash) |
| `zero` baseline with `use_delta_target=False` | Raise `ValueError`: `zero` only valid with `--delta_target` |

---

## Tech Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Decoder output activation for Δρ | Linear (no activation) | Δρ is small-magnitude (‖Δρ‖ < 0.3 daily) and bounded only by construction; tanh clips valid shocks |
| Reconstruction at eval time | ρ̂_{t+1} = ρ_t + Δρ̂ using exact ρ_t | Matches oracle-access baselines at test time; legitimate because the reconstruction uses observed data, not predictions |
| New baselines name | `zero`, `delta_ewma` (not `delta_persistence`) | `zero` is the conceptually correct name — it's the mean-reversion prior, not a persistence model |
| Backward compatibility | `use_delta_target=False` default | All existing experiments, cached results, and eval scripts remain unchanged |
| Shared `corr_today` access | Use existing lookup in `run_split` loop | Already available; no new data preparation pass needed |

---

## Validation Strategy

The `zero` baseline provides a critical sanity check:

> **Invariant**: `zero.mae_reconstructed` == `persistence.mae` (both predict ρ̂_{t+1} = ρ_t).

If this equality does not hold within floating-point tolerance, the delta construction
or reconstruction is incorrect. This check can be automated as an assertion in the eval script.
