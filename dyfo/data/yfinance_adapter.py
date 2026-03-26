"""yfinance adapter — downloads price history, fundamentals, and corporate actions.

All download functions include retry logic with exponential backoff for resilience
against transient network/API failures.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BACKOFF_BASE = 2.0  # seconds


def _retry(fn, description: str, max_retries: int = MAX_RETRIES):
    """Execute fn() with retries and exponential backoff."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = BACKOFF_BASE ** attempt
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.0fs",
                description, attempt + 1, max_retries, e, wait,
            )
            time.sleep(wait)


def download_prices(
    tickers: List[str],
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers.

    Returns
    -------
    pd.DataFrame
        Columns = tickers, index = DatetimeIndex (business-day).
    """
    data = _retry(
        lambda: yf.download(
            tickers,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
        ),
        f"download_prices({len(tickers)} tickers)",
    )
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})
    prices = prices.dropna(how="all")
    return prices


def download_ohlcv(
    tickers: List[str],
    start: str,
    end: str,
) -> Dict[str, pd.DataFrame]:
    """Download full OHLCV for each ticker individually.

    Returns dict mapping ticker -> DataFrame with columns
    [Open, High, Low, Close, Volume].
    """
    result: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            hist = _retry(
                lambda t=tk: t.history(start=start, end=end, auto_adjust=True),
                f"OHLCV({ticker})",
            )
            if not hist.empty:
                result[ticker] = hist[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            logger.warning("Failed to download OHLCV for %s after %d retries", ticker, MAX_RETRIES)
    return result


def get_ticker_info(tickers: List[str]) -> Dict[str, dict]:
    """Fetch .info metadata for each ticker (sector, marketCap, etc.).

    Returns dict mapping ticker -> info dict.
    """
    info: Dict[str, dict] = {}
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            raw = _retry(lambda t=tk: t.info, f"info({ticker})")
            info[ticker] = {
                "sector": raw.get("sector", "Unknown"),
                "market_cap": raw.get("marketCap"),
                "beta": raw.get("beta"),
                "short_name": raw.get("shortName", ticker),
            }
        except Exception:
            logger.warning("Failed to fetch info for %s after %d retries", ticker, MAX_RETRIES)
            info[ticker] = {"sector": "Unknown", "market_cap": None, "beta": None}
    return info


def get_earnings_dates(
    tickers: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch earnings announcement dates and EPS surprise.

    Returns DataFrame with columns [ticker, date, eps_estimate, eps_actual, surprise].
    """
    rows = []
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            ed = _retry(lambda t=tk: t.earnings_dates, f"earnings({ticker})")
            if ed is None or ed.empty:
                continue
            for dt, row in ed.iterrows():
                ts = pd.Timestamp(dt)
                if ts.tzinfo is not None:
                    ts = ts.tz_localize(None)
                if start and ts < pd.Timestamp(start):
                    continue
                if end and ts > pd.Timestamp(end):
                    continue
                rows.append(
                    {
                        "ticker": ticker,
                        "date": ts,
                        "eps_estimate": row.get("EPS Estimate"),
                        "eps_actual": row.get("Reported EPS"),
                        "surprise": row.get("Surprise(%)"),
                    }
                )
        except Exception as e:
            logger.warning("Failed to fetch earnings dates for %s after %d retries: %s", ticker, MAX_RETRIES, e)
    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "eps_estimate", "eps_actual", "surprise"])
    return pd.DataFrame(rows)


def get_corporate_actions(
    tickers: List[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch dividends and stock splits.

    Returns DataFrame with columns [ticker, date, action_type, value].
    """
    rows = []
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            actions = _retry(lambda t=tk: t.actions, f"actions({ticker})")
            if actions is None or actions.empty:
                continue
            for dt, row in actions.iterrows():
                ts = pd.Timestamp(dt)
                if ts.tzinfo is not None:
                    ts = ts.tz_localize(None)
                if ts < pd.Timestamp(start) or ts > pd.Timestamp(end):
                    continue
                splits = row.get("Stock Splits")
                dividends = row.get("Dividends")
                if splits is not None and splits != 0:
                    rows.append(
                        {
                            "ticker": ticker,
                            "date": ts,
                            "action_type": "SPLIT",
                            "value": splits,
                        }
                    )
                if dividends is not None and dividends > 0:
                    rows.append(
                        {
                            "ticker": ticker,
                            "date": ts,
                            "action_type": "DIVIDEND",
                            "value": dividends,
                        }
                    )
        except Exception as e:
            logger.warning("Failed to fetch actions for %s after %d retries: %s", ticker, MAX_RETRIES, e)
    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "action_type", "value"])
    return pd.DataFrame(rows)
