"""FRED adapter — downloads macroeconomic series from the Federal Reserve."""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _get_api_key(api_key: Optional[str] = None) -> str:
    """Resolve FRED API key from argument, env var, or .env file."""
    if api_key:
        return api_key
    key = os.environ.get("FRED_API_KEY", "")
    if not key:
        try:
            from dotenv import load_dotenv

            load_dotenv()
            key = os.environ.get("FRED_API_KEY", "")
        except ImportError:
            pass
    if not key:
        raise ValueError(
            "FRED_API_KEY not set. Provide it as argument, env variable, or in .env file."
        )
    return key


def download_fred_series(
    series_map: Dict[str, str],
    start: str,
    end: str,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Download multiple FRED series and return a single DataFrame.

    Parameters
    ----------
    series_map : dict
        Mapping of readable name -> FRED series ID.
        Example: {"fed_funds_rate": "DFF", "vix": "VIXCLS"}
    start, end : str
        Date range.
    api_key : str, optional
        FRED API key. Falls back to env variable.

    Returns
    -------
    pd.DataFrame
        Columns = readable names, index = DatetimeIndex.
    """
    from fredapi import Fred

    key = _get_api_key(api_key)
    fred = Fred(api_key=key)

    frames: Dict[str, pd.Series] = {}
    for name, series_id in series_map.items():
        try:
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            frames[name] = s
        except Exception:
            logger.warning("Failed to download FRED series %s (%s)", name, series_id)

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def detect_macro_events(
    macro_df: pd.DataFrame,
    threshold_std: float = 1.5,
) -> pd.DataFrame:
    """Detect significant macro releases as events.

    An event is flagged when the day-over-day change of a series
    exceeds `threshold_std` standard deviations of its rolling change.

    Returns DataFrame with columns [date, series, value, change, surprise_z].
    """
    rows = []
    for col in macro_df.columns:
        series = macro_df[col].dropna()
        if len(series) < 30:
            continue
        changes = series.diff()
        rolling_std = changes.rolling(60, min_periods=20).std()
        for dt in changes.index:
            chg = changes.get(dt)
            std = rolling_std.get(dt)
            if pd.isna(chg) or pd.isna(std) or std == 0:
                continue
            z = chg / std
            if abs(z) >= threshold_std:
                rows.append(
                    {
                        "date": dt,
                        "series": col,
                        "value": series[dt],
                        "change": chg,
                        "surprise_z": z,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["date", "series", "value", "change", "surprise_z"])
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
