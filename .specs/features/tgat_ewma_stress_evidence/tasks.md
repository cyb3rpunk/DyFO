# TGAT vs EWMA Stress Evidence Tasks

**Design**: `.specs/features/tgat_ewma_stress_evidence/design.md`
**Status**: Draft

---

## Execution Plan

### Phase 1: Runner Foundation (Sequential)

Build the reusable CLI and causal data path before metrics/reporting.

```text
T1 -> T2 -> T3
```

### Phase 2: Evidence Outputs (Sequential)

Metrics depend on the forecast data model; artifacts depend on metrics.

```text
T3 -> T4 -> T5 -> T6
```

### Phase 3: Protocol and Validation (Sequential)

Official protocol hooks and smoke validation come after the runner is complete.

```text
T6 -> T7 -> T8
```

---

## Task Breakdown

### T1: Add Pair/Battery CLI

**What**: Add CLI options for single-pair and fixed stress-battery execution.
**Where**: `scripts/run_spy_vix_covid_compare.py`
**Depends on**: None
**Reuses**: Existing SPY-VIX default constants and argparse pattern.
**Requirement**: [TGAT-EWMA]-01

**Tools**:
- MCP: filesystem
- Skill: tlc-spec-driven

**Done when**:
- [ ] `--mode pair` evaluates one pair.
- [ ] `--mode battery` evaluates the eight default stress pairs.
- [ ] `--pair` and `--pairs` parse comma-separated ticker pairs.
- [ ] SPY-^VIX remains the default pair.
- [ ] Gate check passes: `.\.venv\Scripts\python.exe -m py_compile scripts\run_spy_vix_covid_compare.py`

**Tests**: validation
**Gate**: quick

**Verify**:
```powershell
.\.venv\Scripts\python.exe scripts\run_spy_vix_covid_compare.py --help
```
Expected: help shows `--mode`, `--pair`, `--pairs`, and `--skip_tgat`.

---

### T2: Normalize Data and Build Causal Forecasts

**What**: Implement robust date normalization, rolling-correlation actuals, EWMA, and Persistence for arbitrary pairs.
**Where**: `scripts/run_spy_vix_covid_compare.py`
**Depends on**: T1
**Reuses**: `_download_prices`, `_ewma_prediction`, `_persistence_prediction` from `scripts/event_study_covid.py`.
**Requirement**: [TGAT-EWMA]-01, [TGAT-EWMA]-06

**Tools**:
- MCP: filesystem
- Skill: tlc-spec-driven

**Done when**:
- [ ] Yahoo Finance outputs are converted to timezone-naive `DatetimeIndex`.
- [ ] Rolling correlations use causal one-step-ahead forecasts through shifted baselines.
- [ ] Missing pair columns raise clear errors.
- [ ] VIX is fetched or reused for high-stress masks.
- [ ] Gate check passes: `.\.venv\Scripts\python.exe -m py_compile scripts\run_spy_vix_covid_compare.py`

**Tests**: validation
**Gate**: quick

**Verify**:
```powershell
.\.venv\Scripts\python.exe scripts\run_spy_vix_covid_compare.py --skip_tgat --pair SPY,^VIX --results_dir results\stress_event_smoke_pair --out_dir figures\stress_event_smoke_pair
```
Expected: command completes and writes one pair report without TGAT training.

---

### T3: Add Optional TGAT Prediction Provider

**What**: Load existing TGAT prediction CSVs or train/save TGAT predictions only when requested.
**Where**: `scripts/run_spy_vix_covid_compare.py`
**Depends on**: T2
**Reuses**: `_load_dyfo_preds`, `train_link_prediction(save_preds_path=...)`, `load_or_prepare_data`.
**Requirement**: [TGAT-EWMA]-03

**Tools**:
- MCP: filesystem
- Skill: tlc-spec-driven

**Done when**:
- [ ] `--skip_tgat` prevents all TGAT loading/training.
- [ ] `--tgat_preds` loads existing pair rows.
- [ ] Missing TGAT pair rows warn and continue.
- [ ] Training is invoked only when TGAT is requested and no prediction file exists.
- [ ] Gate check passes: `.\.venv\Scripts\python.exe -m py_compile scripts\run_spy_vix_covid_compare.py`

**Tests**: validation
**Gate**: quick

**Verify**:
```powershell
.\.venv\Scripts\python.exe scripts\run_spy_vix_covid_compare.py --tgat_preds results\spy_vix_tgat_preds.csv --pair SPY,^VIX --results_dir results\stress_event_smoke_tgat --out_dir figures\stress_event_smoke_tgat
```
Expected: output includes `tgat` metrics for SPY-^VIX or a non-fatal warning if rows are absent.

---

### T4: Implement Per-Period Metrics

**What**: Compute global, stress-window, lag, and directional metrics for each available model.
**Where**: `scripts/run_spy_vix_covid_compare.py`
**Depends on**: T3
**Reuses**: COVID date constants from `scripts/event_study_covid.py`.
**Requirement**: [TGAT-EWMA]-02, [TGAT-EWMA]-06

**Tools**:
- MCP: filesystem
- Skill: tlc-spec-driven

**Done when**:
- [ ] Metrics include `MAE`, `MSE`, `R2`, `Spearman`, and `directional_accuracy`.
- [ ] Periods include `full_test`, `pre_crash`, `covid_crash`, `post_crash`, and `high_vix_days`.
- [ ] Stress metrics include `turning_point_delay_days`, `stress_mae`, `lag_to_threshold`, and `event_window_win`.
- [ ] Undefined Spearman or lag values return unavailable/null rather than crashing.
- [ ] Gate check passes: `.\.venv\Scripts\python.exe -m py_compile scripts\run_spy_vix_covid_compare.py`

**Tests**: validation
**Gate**: quick

**Verify**:
```powershell
.\.venv\Scripts\python.exe scripts\run_spy_vix_covid_compare.py --skip_tgat --pair SPY,BTC-USD --results_dir results\stress_event_smoke_pair --out_dir figures\stress_event_smoke_pair
```
Expected: pair metrics JSON contains all named periods and metrics.

---

### T5: Write Per-Pair Artifacts and Figures

**What**: Save per-pair predictions CSV, metrics JSON, and event-study figure files.
**Where**: `scripts/run_spy_vix_covid_compare.py`
**Depends on**: T4
**Reuses**: Existing event-study plotting constants and colors.
**Requirement**: [TGAT-EWMA]-01, [TGAT-EWMA]-02

**Tools**:
- MCP: filesystem
- Skill: tlc-spec-driven

**Done when**:
- [ ] Each pair writes `<PAIR>_predictions.csv`.
- [ ] Each pair writes `<PAIR>_metrics.json`.
- [ ] Each pair writes `stress_event_compare_<PAIR>.pdf` and `.png`.
- [ ] Figure title reflects conservative stress-adaptation framing.
- [ ] Gate check passes: `.\.venv\Scripts\python.exe -m py_compile scripts\run_spy_vix_covid_compare.py`

**Tests**: validation
**Gate**: quick

**Verify**:
```powershell
.\.venv\Scripts\python.exe scripts\run_spy_vix_covid_compare.py --mode battery --skip_tgat --results_dir results\stress_event_smoke_battery --out_dir figures\stress_event_smoke_battery
```
Expected: eight metrics JSONs, eight predictions CSVs, and eight PDF/PNG figure pairs are created.

---

### T6: Add Aggregate JSON/Markdown Report

**What**: Aggregate pair-level evidence into narrative-ready JSON and Markdown reports.
**Where**: `scripts/run_spy_vix_covid_compare.py`
**Depends on**: T5
**Reuses**: Per-pair metrics dictionary.
**Requirement**: [TGAT-EWMA]-02, [TGAT-EWMA]-05

**Tools**:
- MCP: filesystem
- Skill: tlc-spec-driven

**Done when**:
- [ ] `stress_event_summary.json` is written.
- [ ] `stress_event_report.md` is written.
- [ ] Report states EWMA is a strong smooth autoregressive baseline.
- [ ] Report distinguishes stress adaptation from global R2 dominance.
- [ ] Gate check passes: `.\.venv\Scripts\python.exe -m py_compile scripts\run_spy_vix_covid_compare.py`

**Tests**: validation
**Gate**: quick

**Verify**:
```powershell
Get-Content results\stress_event_smoke_battery\stress_event_report.md
```
Expected: report includes aggregate counts, pair table, and conservative claim framing.

---

### T7: Add Official Walk-Forward Protocol Hook

**What**: Add print/run hooks for the official S&P 50 TGAT vs EWMA vs Persistence command.
**Where**: `scripts/run_spy_vix_covid_compare.py`
**Depends on**: T6
**Reuses**: `scripts/run_bootstrap_eval_temporal_kg_rev3.py`.
**Requirement**: [TGAT-EWMA]-04

**Tools**:
- MCP: filesystem
- Skill: tlc-spec-driven

**Done when**:
- [ ] `--print_walk_forward_command` prints the canonical command.
- [ ] `--run_walk_forward_protocol` executes the same command.
- [ ] The command includes only `tgat`, `ewma`, and `persistence`.
- [ ] The command uses `--n_tickers 50 --step_days 125`.
- [ ] Gate check passes: `.\.venv\Scripts\python.exe -m py_compile scripts\run_spy_vix_covid_compare.py`

**Tests**: validation
**Gate**: quick

**Verify**:
```powershell
.\.venv\Scripts\python.exe scripts\run_spy_vix_covid_compare.py --print_walk_forward_command --skip_tgat --pair SPY,^VIX --results_dir results\stress_event_smoke_final --out_dir figures\stress_event_smoke_final
```
Expected: printed command is `python scripts/run_bootstrap_eval_temporal_kg_rev3.py --variants tgat ewma persistence --n_tickers 50 --step_days 125`.

---

### T8: Validate Delta-Target Exclusion and Smoke Outputs

**What**: Run final checks proving the SDD evidence path excludes delta-target artifacts and the smoke tests pass.
**Where**: `scripts/run_spy_vix_covid_compare.py`, generated smoke output directories.
**Depends on**: T7
**Reuses**: Existing `.venv` and PowerShell commands.
**Requirement**: [TGAT-EWMA]-04, [TGAT-EWMA]-06

**Tools**:
- MCP: filesystem
- Skill: tlc-spec-driven

**Done when**:
- [ ] `rg -n "delta_rho_target|delta_ewma|zero" scripts\run_spy_vix_covid_compare.py` returns no matches.
- [ ] Py compile passes.
- [ ] Pair smoke test passes.
- [ ] Battery smoke test passes.
- [ ] TGAT CSV reuse smoke test passes for `results\spy_vix_tgat_preds.csv`.
- [ ] Temporary smoke output directories are removed or intentionally ignored.

**Tests**: validation
**Gate**: full smoke

**Verify**:
```powershell
rg -n "delta_rho_target|delta_ewma|zero" scripts\run_spy_vix_covid_compare.py
.\.venv\Scripts\python.exe -m py_compile scripts\run_spy_vix_covid_compare.py
```
Expected: `rg` has no matches; compile exits with code 0.

---

## Parallel Execution Map

```text
Phase 1 (Sequential):
  T1 -> T2 -> T3

Phase 2 (Sequential):
  T3 -> T4 -> T5 -> T6

Phase 3 (Sequential):
  T6 -> T7 -> T8
```

**Parallelism constraint**: Tasks are intentionally sequential because all implementation work touches the same script and the validation gates reuse network-backed Yahoo Finance downloads.

---

## Task Granularity Check

| Task | Scope | Status |
| ---- | ----- | ------ |
| T1: Add Pair/Battery CLI | One CLI surface | OK |
| T2: Normalize Data and Build Causal Forecasts | One data-prep/forecast layer | OK |
| T3: Add Optional TGAT Prediction Provider | One TGAT provider layer | OK |
| T4: Implement Per-Period Metrics | One metrics engine | OK |
| T5: Write Per-Pair Artifacts and Figures | One artifact/figure layer | OK |
| T6: Add Aggregate JSON/Markdown Report | One reporting layer | OK |
| T7: Add Official Walk-Forward Protocol Hook | One protocol hook | OK |
| T8: Validate Delta-Target Exclusion and Smoke Outputs | One validation pass | OK |

## Diagram-Definition Cross-Check

| Task | Depends On (task body) | Diagram Shows | Status |
| ---- | ---------------------- | ------------- | ------ |
| T1 | None | None | Match |
| T2 | T1 | T1 -> T2 | Match |
| T3 | T2 | T2 -> T3 | Match |
| T4 | T3 | T3 -> T4 | Match |
| T5 | T4 | T4 -> T5 | Match |
| T6 | T5 | T5 -> T6 | Match |
| T7 | T6 | T6 -> T7 | Match |
| T8 | T7 | T7 -> T8 | Match |

## Test Co-location Validation

The project testing doc defines this area as research/evaluation scripting, where validation commands and artifact checks are the relevant gates rather than unit tests.

| Task | Code Layer Created/Modified | Matrix Requires | Task Says | Status |
| ---- | --------------------------- | --------------- | --------- | ------ |
| T1 | Evaluation script CLI | validation | validation | OK |
| T2 | Evaluation script data path | validation | validation | OK |
| T3 | Evaluation script TGAT provider | validation | validation | OK |
| T4 | Evaluation script metrics | validation | validation | OK |
| T5 | Evaluation script artifacts/figures | validation | validation | OK |
| T6 | Evaluation script reporting | validation | validation | OK |
| T7 | Evaluation script protocol hook | validation | validation | OK |
| T8 | Evidence validation | validation | validation | OK |
