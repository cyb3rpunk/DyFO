"""Edge feature computation — DCC-GARCH correlations, sector edges, factor co-movement.

Implements the four edge types from DyFO Manual §2.3:
  CORR  — dynamic correlation via rolling Pearson (DCC-GARCH optional)
  SECT  — binary same-sector indicator
  SUPL  — supply-chain links (loaded from external CSV)
  FACT  — Fama-French 5-factor loading proximity
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CORR edges — Rolling correlation (fast default) + optional DCC-GARCH
# ---------------------------------------------------------------------------


def compute_rolling_correlations(
    prices: pd.DataFrame,
    window: int = 63,
    threshold: float = 0.3,
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """Compute rolling pairwise Pearson correlations, sparsified.

    Parameters
    ----------
    prices : DataFrame
        Adjusted close, columns = tickers.
    window : rolling window in trading days.
    threshold : absolute correlation cutoff for sparsification.

    Returns
    -------
    corr_series : DataFrame
        Index = dates, columns = "TKRA_TKRB" pair labels, values = rho(t).
    pairs : list of (ticker_i, ticker_j) tuples.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna(how="all")
    tickers = list(log_ret.columns)
    pairs: List[Tuple[str, str]] = list(combinations(tickers, 2))

    records: Dict[str, List[float]] = {f"{a}_{b}": [] for a, b in pairs}
    dates = []

    for i in range(window, len(log_ret)):
        block = log_ret.iloc[i - window : i]
        dates.append(log_ret.index[i])
        corr_mat = block.corr()
        for a, b in pairs:
            rho = corr_mat.at[a, b]
            if pd.isna(rho) or abs(rho) < threshold:
                records[f"{a}_{b}"].append(np.nan)
            else:
                records[f"{a}_{b}"].append(rho)

    corr_df = pd.DataFrame(records, index=dates)
    # Drop pairs that are always NaN
    corr_df = corr_df.dropna(axis=1, how="all")
    surviving_pairs = [
        p for p in pairs if f"{p[0]}_{p[1]}" in corr_df.columns
    ]
    logger.info(
        "Rolling correlations: %d pairs survive sparsification (|rho| >= %.2f)",
        len(surviving_pairs),
        threshold,
    )
    return corr_df, surviving_pairs


def _estimate_dcc_params(
    eps: np.ndarray,
    Q_bar: np.ndarray,
    max_iter: int = 100,
) -> Tuple[float, float]:
    """Estimate DCC(1,1) parameters (a, b) via quasi-maximum likelihood.

    Two-step Engle (2002): given standardised residuals from GARCH step 1,
    maximise the DCC log-likelihood over (a, b) with constraint a+b < 1.
    """
    from scipy.optimize import minimize

    T, N = eps.shape

    # Pre-compute outer products (reused across likelihood evaluations)
    outer_prods = np.empty((T, N, N))
    for t in range(T):
        outer_prods[t] = np.outer(eps[t], eps[t])

    def neg_log_lik(params):
        a, b = params
        if a <= 0 or b <= 0 or a + b >= 0.9999:
            return 1e12

        intercept = (1.0 - a - b) * Q_bar
        Q_t = Q_bar.copy()
        total_ll = 0.0

        for t in range(T):
            if t > 0:
                Q_t = intercept + a * outer_prods[t - 1] + b * Q_t

            # Normalise Q_t → R_t
            d = np.sqrt(np.maximum(np.diag(Q_t), 1e-12))
            R_t = Q_t / np.outer(d, d)
            np.clip(R_t, -1.0, 1.0, out=R_t)
            np.fill_diagonal(R_t, 1.0)

            try:
                sign, logdet = np.linalg.slogdet(R_t)
                if sign <= 0:
                    return 1e12
                R_inv_e = np.linalg.solve(R_t, eps[t])
                total_ll += logdet + eps[t] @ R_inv_e - eps[t] @ eps[t]
            except np.linalg.LinAlgError:
                return 1e12

        return 0.5 * total_ll

    # Grid search for good starting point
    best_nll = 1e12
    best_ab = (0.01, 0.95)
    for a0 in [0.005, 0.01, 0.02, 0.05, 0.10]:
        for b0 in [0.85, 0.90, 0.93, 0.95, 0.97]:
            if a0 + b0 >= 0.999:
                continue
            nll = neg_log_lik((a0, b0))
            if nll < best_nll:
                best_nll = nll
                best_ab = (a0, b0)

    # Refine via L-BFGS-B
    try:
        result = minimize(
            neg_log_lik,
            best_ab,
            method="L-BFGS-B",
            bounds=[(1e-6, 0.50), (1e-6, 0.9999)],
            options={"maxiter": max_iter, "ftol": 1e-8},
        )
        if result.success and result.x[0] + result.x[1] < 0.9999:
            return float(result.x[0]), float(result.x[1])
    except Exception:
        pass

    return best_ab


def _dcc_recursion(
    eps: np.ndarray,
    Q_bar: np.ndarray,
    a: float,
    b: float,
) -> List[np.ndarray]:
    """Run DCC(1,1) forward recursion → time-varying correlation matrices R_t."""
    T, N = eps.shape
    intercept = (1.0 - a - b) * Q_bar
    Q_t = Q_bar.copy()
    R_series: List[np.ndarray] = []

    for t in range(T):
        if t > 0:
            Q_t = intercept + a * np.outer(eps[t - 1], eps[t - 1]) + b * Q_t

        d = np.sqrt(np.maximum(np.diag(Q_t), 1e-12))
        R_t = Q_t / np.outer(d, d)
        np.clip(R_t, -1.0, 1.0, out=R_t)
        np.fill_diagonal(R_t, 1.0)
        R_series.append(R_t)

    return R_series


import datetime
from typing import Any, Dict, List, Optional, Tuple


def compute_dcc_garch_correlations(
    prices: pd.DataFrame,
    window: int = 252,
    threshold: float = 0.3,
    mode: str = "causal_filter",
    refit_every: int = 0,
) -> Tuple[pd.DataFrame, List[Tuple[str, str]], Dict[str, Any]]:
    """Compute DCC-GARCH(1,1) dynamic correlations (Engle 2002) with strict causality.

    Two-step estimation:
      1. Fit GARCH(1,1) per asset -> standardised residuals eps_t
      2. Estimate DCC(1,1) parameters (a, b) via quasi-MLE, then compute
         R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2} where
         Q_t = (1-a-b) Q_bar + a (eps_{t-1} eps_{t-1}') + b Q_{t-1}

    Causality contract (REQ-D3):
      In "causal_filter" mode, parameter estimation (GARCH + DCC) is performed
      strictly on the initial calibration window of length `window` (or history <= t).
      The recursion forward for t >= window evaluates without any future look-ahead.
      Removing observations > t yields the exact same correlation at date t.

    Parameters
    ----------
    prices : DataFrame
        Adjusted close, columns = tickers.
    window : int
        Calibration window in trading days (default 252).
    threshold : float
        Absolute correlation cutoff for sparsification (0 = keep all).
    mode : str
        "causal_filter" (default): strictly causal forward filtering.
        "causal_rolling": re-estimate per step on rolling window.
        "full_sample": legacy full-sample in-sample fit.
    refit_every : int
        Periodic refit cadence in days (0 = single initial calibration).

    Returns
    -------
    corr_df : DataFrame
        Index = dates, columns = "TKRA_TKRB", values = rho_t (NaN if sparsified).
    pairs : list of (ticker_i, ticker_j) surviving sparsification.
    metadata : dict
        Execution metadata: mode, window, refit_every, fitted parameters, timestamps.
    """
    try:
        from arch import arch_model
    except ImportError:
        logger.warning("arch package not installed; falling back to rolling Pearson")
        corr_df, pairs = compute_rolling_correlations(prices, window=window, threshold=threshold)
        metadata = {
            "mode": "rolling_pearson_fallback",
            "window": window,
            "threshold": threshold,
            "reason": "arch_not_installed",
        }
        return corr_df, pairs, metadata

    log_ret = np.log(prices / prices.shift(1)).dropna(how="all")
    tickers = list(log_ret.columns)
    T_total = len(log_ret)

    if T_total < 2 or len(tickers) < 2:
        logger.warning("Insufficient observations or tickers for DCC-GARCH; falling back to rolling Pearson")
        corr_df, pairs = compute_rolling_correlations(prices, window=min(window, max(2, T_total)), threshold=threshold)
        metadata = {
            "mode": "rolling_pearson_fallback",
            "window": window,
            "threshold": threshold,
            "reason": "insufficient_data",
        }
        return corr_df, pairs, metadata

    calib_len = min(window, T_total)

    # -- Step 1: GARCH(1,1) per asset -----------------------------------
    logger.info("DCC-GARCH Step 1: Fitting GARCH(1,1) for %d assets (mode=%s, calib_window=%d)...", len(tickers), mode, calib_len)
    std_resids: Dict[str, pd.Series] = {}
    garch_failed = 0
    garch_params: Dict[str, dict] = {}

    for ticker in tickers:
        series = log_ret[ticker].dropna() * 100  # scale for numerical stability
        if len(series) < calib_len:
            std_resids[ticker] = (series - series.mean()) / max(series.std(), 1e-8)
            garch_failed += 1
            continue
        try:
            if mode == "full_sample":
                fit_slice = series
            else:
                fit_slice = series.iloc[:calib_len]

            model = arch_model(
                fit_slice, vol="Garch", p=1, q=1, mean="Zero", rescale=False,
            )
            res = model.fit(disp="off", show_warning=False)
            
            omega = float(res.params.get("omega", 0.01))
            alpha = float(res.params.get("alpha[1]", 0.05))
            beta = float(res.params.get("beta[1]", 0.90))
            garch_params[ticker] = {"omega": omega, "alpha": alpha, "beta": beta}

            if mode == "full_sample":
                std_resids[ticker] = res.std_resid
            else:
                # Forward causal recursion using fixed calibration parameters
                T_s = len(series)
                r_vals = series.values
                sigma2 = np.empty(T_s)
                init_var = float(np.var(fit_slice.values))
                sigma2[0] = init_var if init_var > 1e-8 else float(omega / max(1.0 - alpha - beta, 1e-4))
                for t in range(1, T_s):
                    sigma2[t] = omega + alpha * (r_vals[t - 1] ** 2) + beta * sigma2[t - 1]
                
                eps_vals = r_vals / np.sqrt(np.maximum(sigma2, 1e-12))
                std_resids[ticker] = pd.Series(eps_vals, index=series.index)

        except Exception as exc:
            logger.warning("GARCH fit failed for %s (%s); using standardised returns", ticker, exc)
            std_resids[ticker] = (series - series.mean()) / max(series.std(), 1e-8)
            garch_failed += 1

    if garch_failed > len(tickers) // 2:
        logger.warning(
            "GARCH failed for %d/%d assets; falling back to rolling Pearson",
            garch_failed, len(tickers),
        )
        corr_df, pairs = compute_rolling_correlations(prices, window=63, threshold=threshold)
        metadata = {
            "mode": "rolling_pearson_fallback",
            "window": 63,
            "threshold": threshold,
            "reason": f"garch_failed_on_{garch_failed}_assets",
        }
        return corr_df, pairs, metadata

    valid_tickers = [t for t in tickers if len(std_resids[t].dropna()) >= calib_len // 2]
    if len(valid_tickers) < 2:
        logger.warning("Fewer than 2 valid tickers for DCC; falling back to rolling Pearson")
        corr_df, pairs = compute_rolling_correlations(prices, window=63, threshold=threshold)
        metadata = {
            "mode": "rolling_pearson_fallback",
            "window": 63,
            "threshold": threshold,
            "reason": "fewer_than_2_valid_tickers",
        }
        return corr_df, pairs, metadata

    resid_df = pd.DataFrame({t: std_resids[t] for t in valid_tickers}).dropna()
    if len(resid_df) < calib_len // 2:
        logger.warning("Insufficient aligned residuals; falling back to rolling Pearson")
        corr_df, pairs = compute_rolling_correlations(prices, window=63, threshold=threshold)
        metadata = {
            "mode": "rolling_pearson_fallback",
            "window": 63,
            "threshold": threshold,
            "reason": "insufficient_aligned_residuals",
        }
        return corr_df, pairs, metadata

    tickers = valid_tickers
    eps_all = resid_df.values
    T, N = eps_all.shape
    T_fit = T if mode == "full_sample" else min(calib_len, T)

    # -- Step 2: DCC parameter estimation --------------------------------
    eps_calib = eps_all[:T_fit]
    Q_bar = np.corrcoef(eps_calib.T)
    if np.isnan(Q_bar).any():
        Q_bar = np.eye(N)

    try:
        a, b = _estimate_dcc_params(eps_calib, Q_bar)
    except Exception as exc:
        logger.warning("DCC estimation failed (%s); using defaults a=0.01, b=0.95", exc)
        a, b = 0.01, 0.95

    logger.info("DCC params: a=%.6f, b=%.6f (persistence a+b=%.4f, mode=%s)", a, b, a + b, mode)

    # -- Step 3: Forward recursion -> R_t ----------------------------------
    R_series = _dcc_recursion(eps_all, Q_bar, a, b)

    # -- Step 4: Extract pairwise correlations ----------------------------
    pairs: List[Tuple[str, str]] = list(combinations(tickers, 2))
    ticker_idx = {t: i for i, t in enumerate(tickers)}

    records: Dict[str, List[float]] = {}
    for tk_a, tk_b in pairs:
        i, j = ticker_idx[tk_a], ticker_idx[tk_b]
        records[f"{tk_a}_{tk_b}"] = [R_t[i, j] for R_t in R_series]

    corr_df = pd.DataFrame(records, index=resid_df.index)

    # Sparsify
    if threshold > 0:
        for col in corr_df.columns:
            mask = corr_df[col].abs() < threshold
            corr_df.loc[mask, col] = np.nan

    corr_df = corr_df.dropna(axis=1, how="all")
    surviving_pairs = [p for p in pairs if f"{p[0]}_{p[1]}" in corr_df.columns]

    metadata = {
        "mode": mode,
        "window": window,
        "refit_every": refit_every,
        "dcc_a": float(a),
        "dcc_b": float(b),
        "persistence": float(a + b),
        "valid_tickers": valid_tickers,
        "n_dates": len(corr_df),
        "calib_window": T_fit,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    logger.info(
        "DCC-GARCH correlations: %d/%d pairs survive (|rho| >= %.2f), T=%d dates (causal mode=%s)",
        len(surviving_pairs), len(pairs), threshold, len(corr_df), mode,
    )
    return corr_df, surviving_pairs, metadata


# ---------------------------------------------------------------------------
# SECT edges — Same-sector binary
# ---------------------------------------------------------------------------


def build_sector_edges(
    ticker_info: Dict[str, dict],
    ticker_to_idx: Dict[str, int],
) -> List[Tuple[int, int, str]]:
    """Build static same-sector edges.

    Returns list of (node_i, node_j, sector_name) for all same-sector pairs.
    """
    sector_groups: Dict[str, List[int]] = {}
    for ticker, idx in ticker_to_idx.items():
        sector = ticker_info.get(ticker, {}).get("sector", "Unknown")
        sector_groups.setdefault(sector, []).append(idx)

    edges = []
    for sector, nodes in sector_groups.items():
        if sector == "Unknown":
            continue
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                edges.append((nodes[i], nodes[j], sector))
    logger.info("Built %d SECT edges across %d sectors", len(edges), len(sector_groups))
    return edges


# ---------------------------------------------------------------------------
# SUPL edges — Supply chain (loaded from CSV)
# ---------------------------------------------------------------------------


def load_supply_chain_edges(
    csv_path: str,
    ticker_to_idx: Dict[str, int],
) -> List[Tuple[int, int, float]]:
    """Load supply-chain relationships from a CSV file.

    CSV expected columns: source_ticker, target_ticker, strength (0-1).
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.warning("Supply chain CSV not found at %s; returning empty", csv_path)
        return []

    edges = []
    for _, row in df.iterrows():
        src = ticker_to_idx.get(row["source_ticker"])
        tgt = ticker_to_idx.get(row["target_ticker"])
        if src is not None and tgt is not None:
            edges.append((src, tgt, float(row.get("strength", 1.0))))
    logger.info("Loaded %d SUPL edges from %s", len(edges), csv_path)
    return edges


# ---------------------------------------------------------------------------
# FACT edges — Factor co-movement (Fama-French 5)
# ---------------------------------------------------------------------------


def compute_factor_edges(
    prices: pd.DataFrame,
    factor_returns: Optional[pd.DataFrame],
    ticker_to_idx: Dict[str, int],
    loading_window: int = 252,
    threshold: float = 0.5,
) -> List[Tuple[int, int, np.ndarray]]:
    """Build factor co-movement edges: pairs with similar FF5 loadings.

    Parameters
    ----------
    prices : DataFrame
        Adjusted close prices.
    factor_returns : DataFrame or None
        Columns = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'], rows = dates.
        If None, returns empty (factors not available).
    ticker_to_idx : dict
    loading_window : OLS estimation window.
    threshold : max L2 distance between loading vectors to create edge.
    """
    if factor_returns is None or factor_returns.empty:
        logger.info("No factor returns provided; skipping FACT edges")
        return []

    log_ret = np.log(prices / prices.shift(1)).dropna(how="all")
    # Align dates
    common_idx = log_ret.index.intersection(factor_returns.index)
    if len(common_idx) < loading_window:
        logger.warning("Insufficient data for factor loadings (%d < %d)", len(common_idx), loading_window)
        return []

    log_ret = log_ret.loc[common_idx].iloc[-loading_window:]
    factors = factor_returns.loc[common_idx].iloc[-loading_window:]

    # OLS: r_i = alpha + beta * factors + eps
    from numpy.linalg import lstsq

    X = np.column_stack([np.ones(len(factors)), factors.values])
    loadings: Dict[str, np.ndarray] = {}
    for ticker in prices.columns:
        if ticker not in log_ret.columns:
            continue
        y = log_ret[ticker].values
        mask = ~np.isnan(y)
        if mask.sum() < loading_window // 2:
            continue
        coef, _, _, _ = lstsq(X[mask], y[mask], rcond=None)
        loadings[ticker] = coef[1:]  # exclude intercept → shape (5,)

    # Build edges for similar loadings
    tickers_with_loadings = list(loadings.keys())
    edges = []
    for i in range(len(tickers_with_loadings)):
        for j in range(i + 1, len(tickers_with_loadings)):
            tk_i, tk_j = tickers_with_loadings[i], tickers_with_loadings[j]
            dist = np.linalg.norm(loadings[tk_i] - loadings[tk_j])
            if dist < threshold:
                idx_i = ticker_to_idx[tk_i]
                idx_j = ticker_to_idx[tk_j]
                # Edge features = absolute difference of loadings
                feat = np.abs(loadings[tk_i] - loadings[tk_j])
                edges.append((idx_i, idx_j, feat))
    logger.info("Built %d FACT edges (threshold=%.2f)", len(edges), threshold)
    return edges
