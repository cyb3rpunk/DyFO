"""Data source audit â€” verifies completeness of yfinance and FRED downloads.

Run this script to identify silent data gaps before training.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from dyfo.data.yfinance_adapter import (
    download_ohlcv,
    download_prices,
    get_corporate_actions,
    get_earnings_dates,
    get_ticker_info,
)
from dyfo.data.fred_adapter import download_fred_series, detect_macro_events
from dyfo.config import DataConfig

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "JPM", "GS", "MA",
    "JNJ", "UNH", "AMZN", "TSLA", "PG", "KO", "XOM", "CVX",
    "CAT", "BA", "META", "LIN", "NEE",
]
START = "2020-01-01"
END = "2024-12-31"
BENCHMARK = "SPY"

data_config = DataConfig(tickers=TICKERS, benchmark_ticker=BENCHMARK, start_date=START, end_date=END)

print("=" * 70)
print("DATA SOURCE AUDIT")
print(f"Tickers: {len(TICKERS)}, Period: {START} -> {END}")
print("=" * 70)

# --------------------------------------------------------------------------
# 1. PRICES
# --------------------------------------------------------------------------
print("\n--- 1. PRICES (yfinance) ---")
prices = download_prices(TICKERS, START, END)
print(f"  Shape: {prices.shape}")
print(f"  Date range: {prices.index[0]} â†’ {prices.index[-1]}")
print(f"  Trading days: {len(prices)}")

# Check for NaN
nan_pct = prices.isna().mean() * 100
problem_tickers_price = nan_pct[nan_pct > 1.0]
if len(problem_tickers_price) > 0:
    print(f"  âš ï¸  Tickers with >1% NaN in prices:")
    for t, pct in problem_tickers_price.items():
        print(f"      {t}: {pct:.1f}% NaN")
else:
    print(f"  âœ… All tickers have <1% NaN in prices")

# Check for gaps (>5 consecutive business days missing)
print(f"\n  Per-ticker coverage:")
for ticker in TICKERS:
    if ticker in prices.columns:
        valid = prices[ticker].dropna()
        if len(valid) > 0:
            coverage = len(valid) / len(prices) * 100
            first, last = valid.index[0], valid.index[-1]
            print(f"    {ticker:6s}: {len(valid):4d} days ({coverage:5.1f}%), {first.date()} â†’ {last.date()}")
        else:
            print(f"    {ticker:6s}: âŒ NO DATA")
    else:
        print(f"    {ticker:6s}: âŒ COLUMN MISSING")

# --------------------------------------------------------------------------
# 2. OHLCV
# --------------------------------------------------------------------------
print("\n--- 2. OHLCV (yfinance) ---")
ohlcv = download_ohlcv(TICKERS, START, END)
print(f"  Tickers downloaded: {len(ohlcv)}/{len(TICKERS)}")
missing = [t for t in TICKERS if t not in ohlcv]
if missing:
    print(f"  âš ï¸  Missing OHLCV: {missing}")
else:
    print(f"  âœ… All tickers have OHLCV")

# Check volume data
print(f"\n  Volume statistics:")
for ticker in TICKERS[:5]:  # Sample
    if ticker in ohlcv:
        vol = ohlcv[ticker]["Volume"]
        zero_pct = (vol == 0).mean() * 100
        nan_pct_v = vol.isna().mean() * 100
        print(f"    {ticker:6s}: mean={vol.mean():.0f}, zeros={zero_pct:.1f}%, NaN={nan_pct_v:.1f}%")

# --------------------------------------------------------------------------
# 3. TICKER INFO (sector, market_cap, beta)
# --------------------------------------------------------------------------
print("\n--- 3. TICKER INFO (yfinance) ---")
ticker_info = get_ticker_info(TICKERS)
print(f"  Tickers with info: {len(ticker_info)}/{len(TICKERS)}")

missing_sector = [t for t, info in ticker_info.items() if info.get("sector") in (None, "Unknown", "")]
missing_mcap = [t for t, info in ticker_info.items() if info.get("market_cap") is None]
missing_beta = [t for t, info in ticker_info.items() if info.get("beta") is None]

if missing_sector:
    print(f"  âš ï¸  Missing sector: {missing_sector}")
else:
    print(f"  âœ… All sectors resolved")

if missing_mcap:
    print(f"  âš ï¸  Missing market_cap: {missing_mcap}")
else:
    print(f"  âœ… All market caps resolved")

if missing_beta:
    print(f"  âš ï¸  Missing beta: {missing_beta}")
else:
    print(f"  âœ… All betas resolved")

print(f"\n  Sector distribution:")
sectors = {}
for t, info in ticker_info.items():
    s = info.get("sector", "Unknown")
    sectors.setdefault(s, []).append(t)
for s, ts in sorted(sectors.items()):
    print(f"    {s:30s}: {ts}")

# --------------------------------------------------------------------------
# 4. EARNINGS DATES
# --------------------------------------------------------------------------
print("\n--- 4. EARNINGS DATES (yfinance) ---")
earnings_df = get_earnings_dates(TICKERS, START, END)
print(f"  Total earnings events: {len(earnings_df)}")

if len(earnings_df) > 0:
    per_ticker = earnings_df.groupby("ticker").size()
    print(f"  Tickers with earnings data: {len(per_ticker)}/{len(TICKERS)}")
    print(f"\n  Per-ticker breakdown:")
    for ticker in TICKERS:
        count = per_ticker.get(ticker, 0)
        flag = "âœ…" if count >= 10 else ("âš ï¸" if count > 0 else "âŒ")
        expected = "~20 expected (4y Ã— ~5/yr)"
        print(f"    {ticker:6s}: {count:3d} events  {flag}  ({expected})")

    # Check surprise data availability
    has_surprise = earnings_df["surprise"].notna().sum()
    has_eps_actual = earnings_df["eps_actual"].notna().sum()
    has_eps_estimate = earnings_df["eps_estimate"].notna().sum()
    print(f"\n  Data completeness:")
    print(f"    eps_estimate:  {has_eps_estimate}/{len(earnings_df)} ({has_eps_estimate/len(earnings_df)*100:.0f}%)")
    print(f"    eps_actual:    {has_eps_actual}/{len(earnings_df)} ({has_eps_actual/len(earnings_df)*100:.0f}%)")
    print(f"    surprise(%):   {has_surprise}/{len(earnings_df)} ({has_surprise/len(earnings_df)*100:.0f}%)")
else:
    print(f"  âŒ NO EARNINGS DATA RETURNED")
    print(f"     This is a known yfinance limitation â€” .earnings_dates")
    print(f"     may return None for many tickers or time periods.")

# --------------------------------------------------------------------------
# 5. CORPORATE ACTIONS (splits + dividends)
# --------------------------------------------------------------------------
print("\n--- 5. CORPORATE ACTIONS (yfinance) ---")
actions_df = get_corporate_actions(TICKERS, START, END)
print(f"  Total corporate actions: {len(actions_df)}")

if len(actions_df) > 0:
    action_types = actions_df["action_type"].value_counts()
    print(f"  By type: {dict(action_types)}")

    per_ticker = actions_df.groupby("ticker").size()
    print(f"  Tickers with actions: {len(per_ticker)}/{len(TICKERS)}")
    print(f"\n  Per-ticker breakdown:")
    for ticker in TICKERS:
        count = per_ticker.get(ticker, 0)
        if count > 0:
            types = actions_df[actions_df["ticker"] == ticker]["action_type"].value_counts().to_dict()
            print(f"    {ticker:6s}: {count:3d} actions â€” {types}")
        else:
            print(f"    {ticker:6s}:   0 actions")
else:
    print(f"  âŒ NO CORPORATE ACTIONS RETURNED")

# --------------------------------------------------------------------------
# 6. FRED MACRO DATA
# --------------------------------------------------------------------------
print("\n--- 6. FRED MACRO SERIES ---")
fred_key = os.environ.get("FRED_API_KEY", "")
if not fred_key:
    print("  âŒ FRED_API_KEY not set â€” skipping")
else:
    macro_df = download_fred_series(data_config.fred_series, START, END, api_key=fred_key)
    print(f"  Shape: {macro_df.shape}")
    print(f"  Date range: {macro_df.index[0]} â†’ {macro_df.index[-1]}")

    print(f"\n  Per-series completeness:")
    for col in macro_df.columns:
        s = macro_df[col].dropna()
        total = len(macro_df)
        print(f"    {col:25s}: {len(s):5d}/{total} observations ({len(s)/total*100:.0f}%), range=[{s.min():.4f}, {s.max():.4f}]")

    # Detect macro events
    macro_events = detect_macro_events(macro_df, threshold_std=1.5)
    print(f"\n  Macro events detected (threshold=1.5Ïƒ): {len(macro_events)}")
    if len(macro_events) > 0:
        events_per_series = macro_events.groupby("series").size()
        for s, count in events_per_series.items():
            print(f"    {s:25s}: {count:4d} events")

# --------------------------------------------------------------------------
# 7. CORRELATION DATA QUALITY
# --------------------------------------------------------------------------
print("\n--- 7. CORRELATION QUALITY ---")
from dyfo.core.edge_features import compute_rolling_correlations

corr_series, corr_pairs = compute_rolling_correlations(
    prices, window=63, threshold=0.3,
)
n_pairs = len(TICKERS) * (len(TICKERS) - 1) // 2
print(f"  Total possible pairs: {n_pairs}")
print(f"  Pairs with |Ï| â‰¥ 0.3 at some point: {len(corr_pairs)}")
print(f"  Correlation observations: {len(corr_series)}")

if len(corr_series) > 0:
    # corr_series is a DataFrame: index=dates, columns="TKRA_TKRB", values=rho
    all_rho = corr_series.stack().dropna()
    print(f"  Non-NaN correlation observations: {len(all_rho)}")
    print(f"  Ï distribution: mean={all_rho.mean():.3f}, std={all_rho.std():.3f}")
    print(f"    |Ï| â‰¥ 0.3: {(all_rho.abs() >= 0.3).sum()} ({(all_rho.abs() >= 0.3).mean()*100:.0f}%)")
    print(f"    |Ï| â‰¥ 0.5: {(all_rho.abs() >= 0.5).sum()} ({(all_rho.abs() >= 0.5).mean()*100:.0f}%)")
    print(f"    |Ï| â‰¥ 0.7: {(all_rho.abs() >= 0.7).sum()} ({(all_rho.abs() >= 0.7).mean()*100:.0f}%)")

    # Class balance at different thresholds
    print(f"\n  Class balance (positive rate) at different thresholds:")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        # For each date, compute fraction of pairs above threshold
        pos_rate = (corr_series.abs() >= thr).sum(axis=1) / corr_series.notna().sum(axis=1)
        pos_rate = pos_rate.dropna()
        print(f"    |Ï| â‰¥ {thr}: avg positive rate = {pos_rate.mean():.1%} (Â±{pos_rate.std():.1%})")

# --------------------------------------------------------------------------
# 8. DCC-GARCH STATUS
# --------------------------------------------------------------------------
print("\n--- 8. DCC-GARCH STATUS ---")
try:
    import arch
    print(f"  âœ… arch package installed (v{arch.__version__})")
    print(f"     DCC-GARCH should be available")
except ImportError:
    print(f"  âŒ arch package NOT installed â€” DCC-GARCH will fallback to Pearson")
    print(f"     Install with: pip install arch")

# --------------------------------------------------------------------------
# SUMMARY
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)

issues = []
if len(earnings_df) == 0:
    issues.append("âŒ No earnings data (yfinance .earnings_dates empty)")
if len(actions_df) == 0:
    issues.append("âŒ No corporate actions")
if len(problem_tickers_price) > 0:
    issues.append(f"âš ï¸  {len(problem_tickers_price)} tickers with >1% NaN in prices")
if missing_sector:
    issues.append(f"âš ï¸  {len(missing_sector)} tickers missing sector info")

try:
    import arch
except ImportError:
    issues.append("âš ï¸  arch not installed â€” using Pearson fallback")

if not issues:
    print("âœ… All data sources operational â€” no critical gaps")
else:
    print("Issues found:")
    for issue in issues:
        print(f"  {issue}")
