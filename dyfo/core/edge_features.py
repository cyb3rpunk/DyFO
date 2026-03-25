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


def compute_dcc_garch_correlations(
    prices: pd.DataFrame,
    window: int = 252,
    threshold: float = 0.3,
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """Compute DCC-GARCH dynamic correlations (requires `arch` package).

    Falls back to rolling Pearson if DCC estimation fails for a pair.
    For large universes (N > 50), this is slow — prefer rolling correlations.
    """
    try:
        from arch import arch_model
    except ImportError:
        logger.warning("arch package not installed; falling back to rolling correlations")
        return compute_rolling_correlations(prices, window=63, threshold=threshold)

    log_ret = np.log(prices / prices.shift(1)).dropna(how="all")
    tickers = list(log_ret.columns)

    # Fit GARCH(1,1) per asset to get standardised residuals
    std_resids: Dict[str, pd.Series] = {}
    for ticker in tickers:
        series = log_ret[ticker].dropna() * 100  # scale for numerical stability
        try:
            model = arch_model(series, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
            res = model.fit(disp="off", show_warning=False)
            std_resids[ticker] = res.std_resid
        except Exception:
            logger.warning("GARCH fit failed for %s; using raw returns", ticker)
            std_resids[ticker] = log_ret[ticker].dropna()

    # Pairwise rolling correlation on standardised residuals
    resid_df = pd.DataFrame(std_resids).dropna(how="any")
    return compute_rolling_correlations(
        resid_df.cumsum().apply(np.exp),  # pseudo-prices for corr
        window=min(window, len(resid_df) // 2),
        threshold=threshold,
    )


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
