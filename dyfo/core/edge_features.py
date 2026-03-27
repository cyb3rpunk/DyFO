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


def compute_dcc_garch_correlations(
    prices: pd.DataFrame,
    window: int = 252,
    threshold: float = 0.3,
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """Compute DCC-GARCH(1,1) dynamic correlations (Engle 2002).

    Two-step estimation:
      1. Fit GARCH(1,1) per asset → standardised residuals ε_t
      2. Estimate DCC(1,1) parameters (a, b) via quasi-MLE, then compute
         R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}  where
         Q_t = (1-a-b) Q̄ + a (ε_{t-1} ε_{t-1}') + b Q_{t-1}

    Falls back to rolling Pearson if GARCH fails for >50 % of assets.

    Parameters
    ----------
    prices : DataFrame
        Adjusted close, columns = tickers.
    window : int
        Minimum number of observations required for GARCH estimation.
    threshold : float
        Absolute correlation cutoff for sparsification (0 = keep all).

    Returns
    -------
    corr_df : DataFrame
        Index = dates, columns = "TKRA_TKRB", values = ρ_t (NaN if sparsified).
    pairs : list of (ticker_i, ticker_j) surviving sparsification.
    """
    try:
        from arch import arch_model
    except ImportError:
        logger.warning("arch package not installed; falling back to rolling Pearson")
        return compute_rolling_correlations(prices, window=63, threshold=threshold)

    log_ret = np.log(prices / prices.shift(1)).dropna(how="all")
    tickers = list(log_ret.columns)

    # -- Step 1: GARCH(1,1) per asset -----------------------------------
    logger.info("DCC-GARCH Step 1: Fitting GARCH(1,1) for %d assets...", len(tickers))
    std_resids: Dict[str, pd.Series] = {}
    garch_failed = 0

    for ticker in tickers:
        series = log_ret[ticker].dropna() * 100  # scale for numerical stability
        if len(series) < window:
            logger.warning(
                "Insufficient data for GARCH on %s (%d < %d obs)",
                ticker, len(series), window,
            )
            std_resids[ticker] = (series - series.mean()) / max(series.std(), 1e-8)
            garch_failed += 1
            continue
        try:
            model = arch_model(
                series, vol="Garch", p=1, q=1, mean="Zero", rescale=False,
            )
            res = model.fit(disp="off", show_warning=False)
            std_resids[ticker] = res.std_resid
        except Exception as exc:
            logger.warning("GARCH fit failed for %s (%s); using standardised returns", ticker, exc)
            std_resids[ticker] = (series - series.mean()) / max(series.std(), 1e-8)
            garch_failed += 1

    if garch_failed > len(tickers) // 2:
        logger.warning(
            "GARCH failed for %d/%d assets; falling back to rolling Pearson",
            garch_failed, len(tickers),
        )
        return compute_rolling_correlations(prices, window=63, threshold=threshold)

    logger.info(
        "GARCH(1,1) fitted: %d OK, %d fallback to standardised returns",
        len(tickers) - garch_failed, garch_failed,
    )

    # Align residuals (common dates, drop any NaN rows)
    # First, drop tickers with insufficient data (e.g. failed downloads)
    valid_tickers = [t for t in tickers if len(std_resids[t].dropna()) >= window // 2]
    dropped = set(tickers) - set(valid_tickers)
    if dropped:
        logger.warning(
            "Dropping %d tickers with insufficient residuals from DCC: %s",
            len(dropped), sorted(dropped),
        )
    if len(valid_tickers) < 2:
        logger.warning("Fewer than 2 valid tickers for DCC; falling back to rolling Pearson")
        return compute_rolling_correlations(prices, window=63, threshold=threshold)

    resid_df = pd.DataFrame({t: std_resids[t] for t in valid_tickers}).dropna()
    if len(resid_df) < window:
        logger.warning(
            "Insufficient aligned residuals (%d < %d); falling back to rolling Pearson",
            len(resid_df), window,
        )
        return compute_rolling_correlations(prices, window=63, threshold=threshold)

    # Update tickers list to only valid ones
    tickers = valid_tickers
    eps = resid_df.values  # (T, N)
    T, N = eps.shape

    # -- Step 2: DCC parameter estimation --------------------------------
    logger.info("DCC-GARCH Step 2: Estimating DCC(1,1) params (T=%d, N=%d)...", T, N)
    Q_bar = np.corrcoef(eps.T)  # unconditional correlation

    try:
        a, b = _estimate_dcc_params(eps, Q_bar)
    except Exception as exc:
        logger.warning("DCC estimation failed (%s); using defaults a=0.01, b=0.95", exc)
        a, b = 0.01, 0.95

    logger.info("DCC params: a=%.6f, b=%.6f (persistence a+b=%.4f)", a, b, a + b)

    # -- Step 3: Forward recursion → R_t ----------------------------------
    logger.info("DCC-GARCH Step 3: Computing time-varying R_t...")
    R_series = _dcc_recursion(eps, Q_bar, a, b)

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

    logger.info(
        "DCC-GARCH correlations: %d/%d pairs survive (|rho| >= %.2f), T=%d dates",
        len(surviving_pairs), len(pairs), threshold, len(corr_df),
    )
    return corr_df, surviving_pairs


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
