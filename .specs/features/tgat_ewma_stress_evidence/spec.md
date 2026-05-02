# TGAT vs EWMA Stress Evidence Specification

## Problem Statement

The current global correlation-forecasting protocol shows EWMA as a very strong baseline because the target is smooth and autoregressive. A credible DyFO/TGAT claim therefore needs to avoid overclaiming global R2 dominance and instead demonstrate where DyFO's event-driven graph representation adds value: stress regimes, regime shifts, lag reduction, cross-asset robustness, and downstream financial utility.

## Goals

- [ ] Generalize the COVID comparison workflow beyond SPY-VIX while preserving SPY-VIX as the canonical default.
- [ ] Produce per-pair CSV/JSON outputs and an aggregate report for stress-window evidence.
- [ ] Provide an official S&P 50 walk-forward command for `tgat`, `ewma`, and `persistence`.
- [ ] Exclude all delta-target work from this evidence track.
- [ ] Make the final claim explicit: DyFO/TGAT is evaluated for stress adaptation and utility, not universal R2 dominance.

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
| ------- | ------ |
| `delta_rho_target`, `zero`, `delta_ewma` variants | The user explicitly asked to ignore this line of work for the TGAT vs EWMA evidence. |
| Claiming TGAT dominates EWMA in global smooth R2 | Existing results show EWMA is structurally strong on smooth DCC-like targets. |
| Training TGAT for every cross-asset pair by default | Too expensive for smoke tests; the battery must run baseline-only quickly and load/train TGAT optionally. |
| Expanding the official universe beyond S&P 50 | S&P 50 remains the primary validated universe; cross-asset pairs are supplementary event studies. |
| Rewriting the DCC-GARCH baseline methodology | `run_wf_dcc_baselines.py` is reused as methodological context, not replaced. |

---

## User Stories

### P1: Cross-Asset Stress Battery MVP

**User Story**: As a researcher, I want to run a fixed stress-event battery across equity, volatility, crypto, bond, gold, and sector ETF pairs so that I can find evidence beyond the single SPY-VIX case.

**Why P1**: A single event-study pair is anecdotal. The battery creates a broader evidence surface while staying fast enough for iteration.

**Acceptance Criteria**:

1. WHEN the user runs `run_spy_vix_covid_compare.py --mode battery --skip_tgat` THEN the system SHALL evaluate `SPY-^VIX`, `SPY-BTC-USD`, `SPY-GLD`, `SPY-TLT`, `QQQ-BTC-USD`, `GLD-BTC-USD`, `XLE-SPY`, and `XLK-SPY`.
2. WHEN the battery completes THEN the system SHALL write one predictions CSV and one metrics JSON per pair.
3. WHEN Yahoo Finance omits metadata for crypto or ETFs THEN the stress battery SHALL continue as a price-only event study.
4. WHEN `--skip_tgat` is passed THEN the system SHALL not train or load TGAT predictions.

**Independent Test**: Run the battery with `--skip_tgat` and verify all eight pair figures plus per-pair CSV/JSON outputs exist.

---

### P1: Honest Metrics and Aggregate Report MVP

**User Story**: As a paper author, I want per-period metrics and a narrative-ready aggregate report so that I can state where EWMA wins and where DyFO/TGAT adds value without overclaiming.

**Why P1**: The core scientific risk is an overbroad claim. The report must frame EWMA as strong while isolating stress-adaptation evidence.

**Acceptance Criteria**:

1. WHEN a pair is evaluated THEN the system SHALL compute `MAE`, `MSE`, `R2`, `Spearman`, and `directional_accuracy` for `full_test`, `pre_crash`, `covid_crash`, `post_crash`, and `high_vix_days`.
2. WHEN a pair is evaluated THEN the system SHALL compute `turning_point_delay_days`, `stress_mae`, `lag_to_threshold`, and `event_window_win` where enough data exists.
3. WHEN multiple pairs are evaluated THEN the system SHALL write `stress_event_summary.json` and `stress_event_report.md`.
4. WHEN TGAT predictions are unavailable THEN aggregate outputs SHALL still include EWMA and Persistence metrics and mark TGAT-specific fields as unavailable rather than failing.

**Independent Test**: Inspect the aggregate report and confirm it includes the honest claim framing plus pair-level EWMA and TGAT columns.

---

### P1: TGAT Prediction Reuse

**User Story**: As an experiment runner, I want to reuse existing TGAT prediction CSVs before training so that validation can be reproducible and cheap.

**Why P1**: Existing `results/spy_vix_tgat_preds.csv` should be usable immediately, and training should be optional.

**Acceptance Criteria**:

1. WHEN `--tgat_preds` points to an existing CSV THEN the system SHALL load matching pair rows from that CSV.
2. WHEN `--tgat_preds` is missing and `--skip_tgat` is false THEN the system SHALL train TGAT for the requested pair and save a prediction CSV.
3. WHEN a prediction CSV does not contain the requested pair THEN the system SHALL warn and continue with baseline metrics.

**Independent Test**: Run `--tgat_preds results/spy_vix_tgat_preds.csv --pair SPY,^VIX` and verify the pair report includes `tgat`.

---

### P2: Official Walk-Forward Protocol Hook

**User Story**: As a reviewer-facing evaluator, I want a canonical command for the official S&P 50 walk-forward comparison so that stress-event evidence can be paired with the established global protocol.

**Why P2**: The stress battery is supplementary; the official walk-forward run anchors the comparison to the validated DyFO evaluation path.

**Acceptance Criteria**:

1. WHEN the user runs `--print_walk_forward_command` THEN the system SHALL print `python scripts/run_bootstrap_eval_temporal_kg_rev3.py --variants tgat ewma persistence --n_tickers 50 --step_days 125`.
2. WHEN the user runs `--run_walk_forward_protocol` THEN the system SHALL execute the same command.
3. WHEN the command is printed or saved in reports THEN it SHALL contain only `tgat`, `ewma`, and `persistence`.

**Independent Test**: Run `--print_walk_forward_command` and verify no delta-target variants appear.

---

### P2: Methodological Context for EWMA Strength

**User Story**: As a manuscript writer, I want the SDD to connect EWMA strength to DCC smoothness so that the paper can explain why EWMA is hard to beat without discrediting DyFO.

**Why P2**: This prevents an interpretability trap where EWMA's high R2 is mistaken for broad financial intelligence.

**Acceptance Criteria**:

1. WHEN the report or manuscript notes are created THEN they SHALL state that EWMA is structurally aligned with smooth autoregressive correlation labels.
2. WHEN the methodology is summarized THEN it SHALL reference `run_wf_dcc_baselines.py` as the leak-free DCC baseline diagnostic.
3. WHEN conclusions are drafted THEN they SHALL distinguish global smooth-label forecasting from stress-event adaptation.

**Independent Test**: Read the generated report and SDD design to verify this framing is present.

---

## Edge Cases

- WHEN a pair has sparse/non-overlapping trading calendars THEN the system SHALL align dates causally and drop missing observations only for the affected metric.
- WHEN Spearman is undefined because one series is constant THEN the system SHALL return unavailable rather than raising.
- WHEN VIX data cannot be downloaded with the pair THEN the system SHALL fetch `^VIX` separately for high-volatility masks.
- WHEN `--skip_tgat` is active THEN no TGAT training side effects SHALL occur.
- WHEN outputs already exist THEN reruns SHALL overwrite only the selected output directory.

---

## Requirement Traceability

| Requirement ID | Story | Phase | Status |
| -------------- | ----- | ----- | ------ |
| [TGAT-EWMA]-01 | P1: Cross-Asset Stress Battery | Tasks | Pending |
| [TGAT-EWMA]-02 | P1: Honest Metrics and Aggregate Report | Tasks | Pending |
| [TGAT-EWMA]-03 | P1: TGAT Prediction Reuse | Tasks | Pending |
| [TGAT-EWMA]-04 | P2: Official Walk-Forward Protocol Hook | Tasks | Pending |
| [TGAT-EWMA]-05 | P2: Methodological Context for EWMA Strength | Tasks | Pending |
| [TGAT-EWMA]-06 | Edge Cases and Causal Alignment | Tasks | Pending |

**Coverage**: 6 total, 6 mapped to tasks, 0 unmapped.

## Success Criteria

- [ ] Battery smoke test generates all eight pair outputs without TGAT training.
- [ ] SPY-VIX run loads `results/spy_vix_tgat_preds.csv` and emits TGAT metrics.
- [ ] Aggregate report includes stress-window metrics and the conservative claim framing.
- [ ] Official walk-forward command contains only `tgat`, `ewma`, and `persistence`.
- [ ] Text search in the feature implementation/report finds no `delta_rho_target`, `zero`, or `delta_ewma`.
