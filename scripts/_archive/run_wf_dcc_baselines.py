#!/usr/bin/env python3
"""
Walk-Forward DCC-GARCH Baseline Evaluation (leak-free).

The current pipeline estimates GARCH(1,1) and DCC parameters on the FULL
dataset (2018-2024) and then applies a forward recursion to generate labels
ρ_t.  This is look-ahead: parameters estimated in 2024 influence the label
for 2020.  EWMA/Persistence exploit this because they predict DCC output
with a DCC-like formula — so artificially smooth labels inflate their R².

This script fixes that by using a STRICTLY WALK-FORWARD estimation:
  For each window:
    1. Fit GARCH(1,1) per asset on TRAINING data only.
    2. Estimate DCC params (a, b, Q̄) on training standardised residuals.
    3. Continue the GARCH + DCC recursion FORWARD into val + test — using
       training parameters but new OOS returns.
    4. Evaluate EWMA and Persistence on these leak-free labels.

The Δ R² (WF minus Full-Sample) isolates the label-smoothness leakage.

Usage
-----
  python scripts/run_wf_dcc_baselines.py
"""

from __future__ import annotations

import glob
import pickle
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Walk-forward protocol (must match bootstrap eval) ────────────────────────
WF_TRAIN   = 375 #500
WF_VAL     = 125
WF_TEST    = 125
WF_STEP    = 125
DATA_START = "2018-01-01"
DATA_END   = "2024-12-31"
N_TICKERS  = 50
EWMA_ALPHA = 0.05

WINDOW_LABELS = [
    "May–Nov\n2020", "Nov20–\nMay21", "May–Oct\n2021",
    "Nov21–\nApr22", "Apr–Oct\n2022", "Oct22–\nApr23",
    "Apr–Sep\n2023",  "Oct23–\nMar24", "Mar–Sep\n2024",
]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_prices() -> pd.DataFrame:
    """Load from prepared-data cache (fast) or yfinance (fallback)."""
    for pkl in sorted(glob.glob(str(ROOT / "results" / "prepared_data_cache_*.pkl"))):
        try:
            with open(pkl, "rb") as f:
                cache = pickle.load(f)
            prices = cache["prices"]
            if len(prices.columns) >= N_TICKERS:
                from dyfo.core.ticker_registry import get_tickers
                keep = [t for t in get_tickers(N_TICKERS) if t in prices.columns][:N_TICKERS]
                print(f"Cache loaded: {pkl}  ({len(prices)} days, {len(keep)} tickers)")
                return prices[keep].dropna(how="all")
        except Exception as exc:
            print(f"  skip {pkl}: {exc}")

    print("Downloading from yfinance …")
    from dyfo.core.ticker_registry import get_tickers
    from dyfo.data.yfinance_adapter import download_prices
    tickers = get_tickers(N_TICKERS)
    return download_prices(tickers, DATA_START, DATA_END)


# ── Walk-forward split builder ───────────────────────────────────────────────

def build_splits(index: pd.DatetimeIndex) -> List[Dict]:
    splits, i = [], 0
    while True:
        te_s = i + WF_TRAIN + WF_VAL
        te_e = te_s + WF_TEST
        if te_e > len(index):
            break
        splits.append({
            "train": index[i            : i + WF_TRAIN],
            "val":   index[i + WF_TRAIN : te_s],
            "test":  index[te_s         : te_e],
        })
        i += WF_STEP
    return splits


# ── GARCH(1,1): fit on training, apply forward on OOS ───────────────────────

def _garch_fit_and_forward(
    r_train: pd.Series,
    r_oos: pd.Series,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return standardised residuals for (train, oos) using training parameters only."""
    from arch import arch_model

    scale = 100.0
    tr = (r_train * scale).dropna()
    os = (r_oos   * scale).fillna(0.0)

    try:
        res = arch_model(tr, vol="Garch", p=1, q=1, mean="Zero", rescale=False).fit(
            disp="off", show_warning=False
        )
        omega = float(res.params["omega"])
        alpha = float(res.params["alpha[1]"])
        beta  = float(res.params["beta[1]"])

        eps_tr = res.std_resid.values.astype(float)

        # Forward GARCH: σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}
        sigma2  = float(res.conditional_volatility.iloc[-1]) ** 2
        r_prev  = float(tr.iloc[-1])
        eps_os  = np.empty(len(os))
        for k, r_t in enumerate(os.values):
            sigma2      = omega + alpha * r_prev**2 + beta * sigma2
            sigma2      = max(sigma2, 1e-8)
            eps_os[k]   = r_t / np.sqrt(sigma2)
            r_prev      = r_t

        return eps_tr, eps_os

    except Exception:
        # Fallback: standardise with training moments
        mu, sd = float(tr.mean()), max(float(tr.std()), 1e-8)
        return ((tr - mu) / sd).values.astype(float), ((os - mu) / sd).values.astype(float)


# ── DCC parameter estimation ─────────────────────────────────────────────────

def _estimate_dcc(eps: np.ndarray, Q_bar: np.ndarray) -> Tuple[float, float]:
    """Quasi-MLE for DCC(1,1) params (a, b) on training residuals."""
    T = len(eps)

    def neg_ll(params: np.ndarray) -> float:
        a, b = params
        if a <= 0 or b <= 0 or a + b >= 1.0:
            return 1e10
        intercept = (1.0 - a - b) * Q_bar
        Q, ll = Q_bar.copy(), 0.0
        for t in range(T):
            e = eps[t]
            if t > 0:
                e_prev = eps[t - 1]
                Q = intercept + a * np.outer(e_prev, e_prev) + b * Q
            dq = np.sqrt(np.maximum(np.diag(Q), 1e-10))
            R  = Q / np.outer(dq, dq)
            np.fill_diagonal(R, 1.0)
            try:
                sgn, ldet = np.linalg.slogdet(R)
                if sgn <= 0:
                    return 1e10
                ll += ldet + e @ np.linalg.solve(R, e) - e @ e
            except Exception:
                return 1e10
        return ll / T

    res = minimize(
        neg_ll, [0.01, 0.95], method="L-BFGS-B",
        bounds=[(1e-6, 0.499), (1e-6, 0.999)],
        options={"maxiter": 200, "ftol": 1e-9},
    )
    return (float(res.x[0]), float(res.x[1])) if res.success else (0.01, 0.95)


# ── Full WF-DCC pipeline for one window ─────────────────────────────────────

def wf_dcc_window(
    prices: pd.DataFrame,
    train_idx: pd.DatetimeIndex,
    oos_idx: pd.DatetimeIndex,   # val + test concatenated
    min_obs: int = 252,
) -> Optional[pd.DataFrame]:
    """
    Fit DCC-GARCH on training data, apply forward to oos_idx.
    Returns correlation DataFrame (index=oos_idx, columns='TKA_TKB').
    Returns None if estimation fails.
    """
    tickers = list(prices.columns)
    log_ret = np.log(prices / prices.shift(1)).dropna(how="all")

    r_train = log_ret.reindex(train_idx).dropna(how="all")
    r_oos   = log_ret.reindex(oos_idx).fillna(0.0)

    if len(r_train) < min_obs:
        return None

    # ── Step 1: GARCH per asset ──────────────────────────────────────────────
    eps_tr_dict: Dict[str, np.ndarray] = {}
    eps_os_dict: Dict[str, np.ndarray] = {}
    valid = []

    for tkr in tickers:
        tr_s = r_train[tkr].dropna()
        os_s = r_oos[tkr]
        if len(tr_s) < min_obs // 2:
            continue
        et, eo = _garch_fit_and_forward(tr_s, os_s)
        # Align lengths
        eps_tr_dict[tkr] = et[-len(r_train):]
        eps_os_dict[tkr] = eo[: len(r_oos)]
        valid.append(tkr)

    if len(valid) < 2:
        return None

    T_tr  = min(len(v) for v in eps_tr_dict.values())
    T_os  = min(len(v) for v in eps_os_dict.values())

    eps_tr = np.column_stack([eps_tr_dict[t][-T_tr:] for t in valid])
    eps_os = np.column_stack([eps_os_dict[t][:T_os]  for t in valid])

    # ── Step 2: DCC estimation on training residuals ─────────────────────────
    Q_bar     = np.corrcoef(eps_tr.T)
    a, b      = _estimate_dcc(eps_tr, Q_bar)

    # ── Step 3: Burn-in Q through training to get Q_T ────────────────────────
    intercept = (1.0 - a - b) * Q_bar
    Q = Q_bar.copy()
    for t in range(T_tr):
        if t > 0:
            e_prev = eps_tr[t - 1]
            Q = intercept + a * np.outer(e_prev, e_prev) + b * Q

    # ── Step 4: DCC forward recursion on OOS ─────────────────────────────────
    pair_cols = [f"{ti}_{tj}" for ti, tj in combinations(valid, 2)]
    t_idx     = {tkr: i for i, tkr in enumerate(valid)}
    buf       = {col: np.empty(T_os) for col in pair_cols}

    for t in range(T_os):
        dq = np.sqrt(np.maximum(np.diag(Q), 1e-10))
        R  = Q / np.outer(dq, dq)
        np.fill_diagonal(R, 1.0)
        np.clip(R, -1.0, 1.0, out=R)
        for ti, tj in combinations(valid, 2):
            buf[f"{ti}_{tj}"][t] = R[t_idx[ti], t_idx[tj]]
        e_t = eps_os[t]
        Q = intercept + a * np.outer(e_t, e_t) + b * Q

    oos_dates = r_oos.index[:T_os]
    return pd.DataFrame(buf, index=oos_dates)


# ── Baseline evaluation on a correlation DataFrame ──────────────────────────

def eval_baselines(
    test_corr: pd.DataFrame,
    warmup_corr: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    Evaluate EWMA and Persistence on test_corr.

    EWMA state is warmed up on warmup_corr (val period) before the test loop,
    matching how train_link_prediction.py inherits temporal state.

    R² formula: 1 - SS_res / SS_tot (pooled across all pairs and test days).
    """
    ewma_preds, pers_preds, targets = [], [], []

    for col in test_corr.columns:
        test_s = test_corr[col].dropna()
        if len(test_s) < 3:
            continue

        # Warm up EWMA on the val-period correlation for this pair
        if warmup_corr is not None and col in warmup_corr.columns:
            wu = warmup_corr[col].dropna()
            if len(wu) > 0:
                ewma_state = float(wu.ewm(alpha=EWMA_ALPHA, adjust=False).mean().iloc[-1])
            else:
                ewma_state = float(test_s.iloc[0])
        else:
            ewma_state = float(test_s.iloc[0])

        vals = test_s.values
        for i in range(len(vals) - 1):
            rho_today    = float(vals[i])
            rho_tomorrow = float(vals[i + 1])

            # EWMA: update state with today, then predict tomorrow
            ewma_state = EWMA_ALPHA * rho_today + (1 - EWMA_ALPHA) * ewma_state

            ewma_preds.append(ewma_state)
            pers_preds.append(rho_today)   # Persistence: ρ̂_{t+1} = ρ_t
            targets.append(rho_tomorrow)

    y      = np.asarray(targets,    dtype=float)
    y_ewma = np.asarray(ewma_preds, dtype=float)
    y_pers = np.asarray(pers_preds, dtype=float)
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    if ss_tot < 1e-12:
        return {"ewma_r2": float("nan"), "persistence_r2": float("nan")}

    r2_ewma = 1.0 - float(np.sum((y - y_ewma) ** 2)) / ss_tot
    r2_pers = 1.0 - float(np.sum((y - y_pers) ** 2)) / ss_tot
    return {"ewma_r2": r2_ewma, "persistence_r2": r2_pers}


# ── Load full-sample results from existing JSON ──────────────────────────────

def load_full_sample() -> Dict[str, List[float]]:
    import json
    path = (
        ROOT / "results"
        / "bootstrap_eval_tkg_rev3_20260501_200449"
        / "bootstrap_summary_tkg_rev3.json"
    )
    if not path.exists():
        print(f"[WARN] Full-sample JSON not found: {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    data = payload["metrics_by_variant"]
    return {
        "ewma_full":        [w["r_squared"] for w in data["ewma"]],
        "persistence_full": [w["r_squared"] for w in data["persistence"]],
        "_run_config":      payload.get("run_config", {}),
    }


# ── Figure ───────────────────────────────────────────────────────────────────

def make_figure(
    wf: Dict[str, List[float]],
    fs: Dict[str, List[float]],
    out_dir: Path,
) -> None:
    plot_data = {k: v for k, v in {**wf, **fs}.items() if not k.startswith("_")}
    n = max(len(v) for v in plot_data.values())
    x = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(
        "Impact of Full-Sample DCC-GARCH Label Leakage on Baseline $R^2$\n"
        r"Walk-Forward (WF): GARCH/DCC fit on training data only  $\;\bullet\;$  "
        "Full-Sample (FS): fit on 2018–2024",
        fontsize=9.5, y=1.01,
    )

    STYLES = {
        "ewma_full":        dict(color="#1565C0", ls="--", lw=1.8, marker="s", ms=5,
                                 label="EWMA — Full-Sample (current)"),
        "ewma_wf":          dict(color="#64B5F6", ls="-",  lw=2.2, marker="o", ms=6,
                                 label="EWMA — Walk-Forward (leak-free)"),
        "persistence_full": dict(color="#424242", ls="--", lw=1.8, marker="^", ms=5,
                                 label="Persistence — Full-Sample (current)"),
        "persistence_wf":   dict(color="#BDBDBD", ls="-",  lw=2.2, marker="D", ms=5,
                                 label="Persistence — Walk-Forward (leak-free)"),
    }

    all_data = plot_data

    for ax, (key_fs, key_wf), title in [
        (axes[0], ("ewma_full", "ewma_wf"), "EWMA"),
        (axes[1], ("persistence_full", "persistence_wf"), "Persistence"),
    ]:
        for key in (key_fs, key_wf):
            if key in all_data:
                vals = all_data[key]
                st   = STYLES[key]
                ax.plot(x[: len(vals)], vals, **st)

        # Annotate mean Δ
        if key_fs in all_data and key_wf in all_data:
            overlap = min(len(all_data[key_fs]), len(all_data[key_wf]))
            fs_mean = np.nanmean(all_data[key_fs][:overlap])
            wf_mean = np.nanmean(all_data[key_wf][:overlap])
            delta   = wf_mean - fs_mean
            sign    = "+" if delta >= 0 else ""
            ax.text(
                0.97, 0.04,
                f"Mean $\\Delta R^2$ = {sign}{delta:.4f}\n"
                f"FS: {fs_mean:.4f}   WF: {wf_mean:.4f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.9),
            )

        labels = WINDOW_LABELS[:n]
        if len(labels) < n:
            labels += [f"W{i}" for i in range(len(labels) + 1, n + 1)]
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("$R^2$", fontsize=10)
        ax.set_ylim(0.6, 1.03)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.25, lw=0.5)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out_dir / f"wf_dcc_baseline_comparison.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved -> {p}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("Walk-Forward DCC-GARCH Baseline Evaluation (leak-free)")
    print("=" * 65)

    prices = load_prices()
    bdays  = pd.bdate_range(DATA_START, DATA_END)
    prices = prices.reindex(prices.index.intersection(bdays)).dropna(how="all")
    print(f"Prices ready: {len(prices)} days, {len(prices.columns)} tickers\n")

    splits = build_splits(prices.index)
    print(f"Walk-forward windows: {len(splits)}\n")

    wf: Dict[str, List[float]] = {"ewma_wf": [], "persistence_wf": []}
    fs = load_full_sample()
    fs_cfg = fs.get("_run_config", {})
    if fs_cfg:
        print(
            "Reference FS config: "
            f"train={fs_cfg.get('train_days', 'na')}d  "
            f"val={fs_cfg.get('val_days', 'na')}d  "
            f"test={fs_cfg.get('test_days', 'na')}d  "
            f"step={fs_cfg.get('step_days', 'na')}d  "
            f"windows={fs_cfg.get('n_windows', 'na')}\n"
        )

    for i, sp in enumerate(splits):
        t0 = time.time()
        te_start = str(sp["test"][0].date())
        te_end   = str(sp["test"][-1].date())
        print(f"W{i+1:02d}  test {te_start} -> {te_end}  "
              f"(train={len(sp['train'])}d  val={len(sp['val'])}d  test={len(sp['test'])}d)")

        # Combine val + test as OOS; val is warmup, test is evaluation
        oos_idx = sp["val"].append(sp["test"])

        corr_df = wf_dcc_window(prices, sp["train"], oos_idx)

        if corr_df is None or corr_df.empty:
            print(f"  WF-DCC failed — skipping")
            wf["ewma_wf"].append(float("nan"))
            wf["persistence_wf"].append(float("nan"))
            continue

        val_corr  = corr_df.reindex(sp["val"])
        test_corr = corr_df.reindex(sp["test"])

        metrics = eval_baselines(test_corr, warmup_corr=val_corr)
        wf["ewma_wf"].append(metrics["ewma_r2"])
        wf["persistence_wf"].append(metrics["persistence_r2"])

        elapsed = time.time() - t0
        print(f"  EWMA R2={metrics['ewma_r2']:.4f}   "
              f"Persistence R2={metrics['persistence_r2']:.4f}   ({elapsed:.0f}s)")

    fs = load_full_sample()

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("SUMMARY  —  Full-Sample (FS) vs Walk-Forward (WF) DCC labels")
    print("=" * 75)
    hdr = f"{'W':>3}  {'EWMA-FS':>9}  {'EWMA-WF':>9}  {'ΔEWMA':>7}  |  " \
          f"{'Pers-FS':>9}  {'Pers-WF':>9}  {'ΔPers':>7}"
    print(hdr)
    print("-" * 75)

    efs_list = fs.get("ewma_full",        [])
    pfs_list = fs.get("persistence_full", [])
    ewf_list = wf["ewma_wf"]
    pwf_list = wf["persistence_wf"]
    overlap_n = min(len(efs_list), len(ewf_list), len(pfs_list), len(pwf_list))

    for j in range(len(ewf_list)):
        efs = efs_list[j] if j < len(efs_list) else float("nan")
        pfs = pfs_list[j] if j < len(pfs_list) else float("nan")
        ewf = ewf_list[j]
        pwf = pwf_list[j]
        print(f"W{j+1:>2}  {efs:>9.4f}  {ewf:>9.4f}  {ewf-efs:>+7.4f}  |  "
              f"{pfs:>9.4f}  {pwf:>9.4f}  {pwf-pfs:>+7.4f}")

    def mean_valid(lst: List[float]) -> float:
        vals = [v for v in lst if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    print("-" * 75)
    efs_m = mean_valid(efs_list[:overlap_n])
    ewf_m = mean_valid(ewf_list[:overlap_n])
    pfs_m = mean_valid(pfs_list[:overlap_n])
    pwf_m = mean_valid(pwf_list[:overlap_n])
    print(f"{'Avg':>3}  {efs_m:>9.4f}  {ewf_m:>9.4f}  {ewf_m-efs_m:>+7.4f}  |  "
          f"{pfs_m:>9.4f}  {pwf_m:>9.4f}  {pwf_m-pfs_m:>+7.4f}")
    if overlap_n < len(ewf_list):
        print(
            f"[note] FS JSON has only {overlap_n} comparable windows; "
            f"WF-only windows are excluded from the mean delta."
        )

    print("\nInterpretation:")
    delta_ewma = ewf_m - efs_m
    if abs(delta_ewma) < 0.01:
        print("  EWMA Δ < 0.01 => full-sample leakage is NOT the main driver of R²=0.99.")
        print("  Structural similarity (EWMA ≈ DCC process) explains the high R².")
    else:
        print(f"  EWMA Δ = {delta_ewma:+.4f} => leakage accounts for {abs(delta_ewma):.4f} R² points.")

    make_figure(wf, fs, ROOT / "figures")


if __name__ == "__main__":
    main()
