"""Fama-French adapter — downloads factor returns from Ken French's data library.

Downloads the Fama-French 5-factor (2x3) daily dataset directly from:
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

No extra dependencies beyond pandas and the standard library.
"""

from __future__ import annotations

import io
import logging
import os
import time
import zipfile
from typing import Optional
from urllib.request import urlopen
from urllib.error import URLError

import pandas as pd

logger = logging.getLogger(__name__)

# Ken French Data Library URLs
FF5_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)

MAX_RETRIES = 3
BACKOFF_BASE = 2.0
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def download_ff5_factors(
    start: str,
    end: str,
    cache: bool = True,
) -> Optional[pd.DataFrame]:
    """Download Fama-French 5-factor daily returns.

    Parameters
    ----------
    start, end : str
        Date range (inclusive).
    cache : bool
        If True, cache the downloaded CSV in data/ff5_daily.csv to avoid
        re-downloading on subsequent runs.

    Returns
    -------
    DataFrame with columns ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF'],
    index = DatetimeIndex, values in decimal (e.g. 0.01 = 1%).
    Returns None if download fails.
    """
    cache_path = os.path.join(CACHE_DIR, "ff5_daily.csv")

    # Try loading from cache first
    if cache and os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            df = df.loc[start:end]
            if not df.empty:
                logger.info(
                    "Loaded FF5 factors from cache (%d days, %s to %s)",
                    len(df), df.index[0].date(), df.index[-1].date(),
                )
                return df
        except Exception:
            logger.warning("Failed to read FF5 cache; re-downloading")

    # Download from Ken French website
    raw_df = _download_ff5_zip()
    if raw_df is None:
        return None

    # Cache for future use
    if cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        raw_df.to_csv(cache_path)
        logger.info("Cached FF5 factors to %s", cache_path)

    # Filter date range
    df = raw_df.loc[start:end]
    logger.info(
        "FF5 factors: %d trading days (%s to %s)",
        len(df), df.index[0].date(), df.index[-1].date(),
    )
    return df


def _download_ff5_zip() -> Optional[pd.DataFrame]:
    """Download and parse the FF5 daily CSV from the zip archive."""
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                "Downloading FF5 factors from Ken French library (attempt %d/%d)...",
                attempt + 1, MAX_RETRIES,
            )
            response = urlopen(FF5_DAILY_URL, timeout=30)
            zip_bytes = response.read()
            break
        except (URLError, TimeoutError, OSError) as e:
            if attempt == MAX_RETRIES - 1:
                logger.warning(
                    "Failed to download FF5 factors after %d retries: %s",
                    MAX_RETRIES, e,
                )
                return None
            wait = BACKOFF_BASE ** attempt
            logger.warning(
                "FF5 download failed (attempt %d/%d): %s — retrying in %.0fs",
                attempt + 1, MAX_RETRIES, e, wait,
            )
            time.sleep(wait)

    # Parse the zip file
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            csv_name = [n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv")]
            if not csv_name:
                logger.warning("No CSV file found in FF5 zip archive")
                return None

            with zf.open(csv_name[0]) as f:
                content = f.read().decode("utf-8")

        return _parse_ff5_csv(content)

    except Exception as e:
        logger.warning("Failed to parse FF5 zip: %s", e)
        return None


def _parse_ff5_csv(content: str) -> pd.DataFrame:
    """Parse the Ken French CSV format (header rows, then data, then annual section)."""
    lines = content.strip().split("\n")

    # Find the start of daily data (first line that starts with a date: YYYYMMDD)
    data_start = None
    for i, line in enumerate(lines):
        stripped = line.strip().split(",")[0].strip()
        if stripped.isdigit() and len(stripped) == 8:
            data_start = i
            break

    if data_start is None:
        raise ValueError("Could not find data start in FF5 CSV")

    # Find the header line (typically one line before data)
    # Read until we hit a non-data row (annual data section or EOF)
    rows = []
    for i in range(data_start, len(lines)):
        parts = [p.strip() for p in lines[i].split(",")]
        if not parts[0].isdigit() or len(parts[0]) != 8:
            break
        rows.append(parts)

    # Build DataFrame
    df = pd.DataFrame(
        rows,
        columns=["date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
    )
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date")

    # Convert from percentage to decimal
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

    df = df.dropna()
    logger.info("Parsed FF5: %d daily observations", len(df))
    return df
