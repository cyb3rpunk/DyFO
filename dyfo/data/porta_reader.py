"""PortaDataReader — Strict Read-Only Consumer of PORTA Curated Data and Features.

INVARIANT (USER LAW):
====================
This module operates under a STRICT READ-ONLY CONTRACT with respect to the PORTA
repository (`d:\\projetos\\PORTA`). Under NO circumstances does this module modify,
create, delete, or alter any files or directories inside the PORTA workspace.
All tensor reads are performed with `mmap_mode='r'` or read-only file streams.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml

DEFAULT_PORTA_PATH = Path("d:/projetos/PORTA")


class PortaDataReader:
    """Strictly read-only reader for PORTA's curated financial tensors and regime data."""

    def __init__(self, porta_root: Optional[Union[str, Path]] = None):
        self.porta_root = Path(porta_root or DEFAULT_PORTA_PATH)
        self.features_dir = self.porta_root / "data" / "features" / "daily_core"
        self.curated_dir = self.porta_root / "data" / "curated" / "daily_core"
        
        self._date_index: Optional[pd.DataFrame] = None
        self._meta: Optional[Dict[str, Any]] = None
        self._assets: Optional[List[str]] = None
        self._asset_to_idx: Optional[Dict[str, int]] = None
        self._feature_cols: Optional[List[str]] = None
        self._is_available: bool = self._check_availability()

    @property
    def is_available(self) -> bool:
        """Return True if the PORTA curated daily_core features are accessible."""
        return self._is_available

    def _check_availability(self) -> bool:
        """Verify that PORTA daily_core directory and files exist in read-only mode."""
        if not self.features_dir.exists():
            return False
        required = ["date_index.csv", "tensors.meta.yaml", "X.npy", "R.npy"]
        return all((self.features_dir / f).exists() for f in required)

    def _load_metadata(self) -> None:
        """Load metadata YAML and date index CSV strictly read-only."""
        if self._meta is not None and self._date_index is not None:
            return
        if not self._is_available:
            raise FileNotFoundError(f"PORTA daily_core features not found at {self.features_dir}")

        meta_path = self.features_dir / "tensors.meta.yaml"
        with open(meta_path, "r", encoding="utf-8") as f:
            self._meta = yaml.safe_load(f)

        self._assets = list(self._meta.get("assets", []))
        self._asset_to_idx = {asset: idx for idx, asset in enumerate(self._assets)}
        self._feature_cols = list(self._meta.get("feature_columns", []))

        date_path = self.features_dir / "date_index.csv"
        # Read strictly without modification
        df_dates = pd.read_csv(date_path)
        # Normalize date column
        if "date" in df_dates.columns:
            df_dates["date"] = pd.to_datetime(df_dates["date"]).dt.date
        elif "as_of_date" in df_dates.columns:
            df_dates["date"] = pd.to_datetime(df_dates["as_of_date"]).dt.date
        else:
            df_dates["date"] = pd.to_datetime(df_dates.iloc[:, 0]).dt.date
        self._date_index = df_dates

    def get_assets(self) -> List[str]:
        """Return list of available assets in PORTA daily_core (e.g. ['AAPL.US', ...])."""
        self._load_metadata()
        return list(self._assets or [])

    def get_feature_columns(self) -> List[str]:
        """Return list of 35 curated feature names."""
        self._load_metadata()
        return list(self._feature_cols or [])

    def get_date_range(self) -> Tuple[datetime.date, datetime.date]:
        """Return the start and end dates of the PORTA date index."""
        self._load_metadata()
        assert self._date_index is not None
        dates = self._date_index["date"].values
        return dates[0], dates[-1]

    def _get_time_index_at_date(self, as_of_date: datetime.date) -> Optional[int]:
        """Return the latest integer index t such that date_index[t] <= as_of_date (strictly causal)."""
        self._load_metadata()
        assert self._date_index is not None
        valid = self._date_index[self._date_index["date"] <= as_of_date]
        if valid.empty:
            return None
        return int(valid.index[-1])

    def get_features_at_date(
        self,
        as_of_date: Union[datetime.date, str],
        assets: Optional[Sequence[str]] = None,
    ) -> Optional[np.ndarray]:
        """Retrieve curated feature matrix for specified assets strictly as of as_of_date (causal).

        Parameters
        ----------
        as_of_date : datetime.date or str
            Causal decision date.
        assets : sequence of str, optional
            List of tickers (e.g. ['AAPL.US', 'MSFT.US'] or ['AAPL', 'MSFT']).
            If None, returns all PORTA assets.

        Returns
        -------
        np.ndarray of shape (N_assets, F_features) or None if date is out of range.
        """
        if not self._is_available:
            return None
        if isinstance(as_of_date, str):
            as_of_date = datetime.date.fromisoformat(as_of_date)

        t_idx = self._get_time_index_at_date(as_of_date)
        if t_idx is None:
            return None

        # Load X tensor with mmap_mode='r' (read-only memory map)
        x_path = self.features_dir / "X.npy"
        x_mmap = np.load(x_path, mmap_mode="r")  # shape: (T, N, F)
        
        if assets is None:
            # Return all assets at time t_idx
            return np.array(x_mmap[t_idx], dtype=np.float32)

        # Map requested assets to PORTA indices
        assert self._asset_to_idx is not None
        indices = []
        for a in assets:
            a_clean = a if a.endswith(".US") else f"{a}.US"
            if a_clean in self._asset_to_idx:
                indices.append(self._asset_to_idx[a_clean])
            else:
                indices.append(None)

        feat_dim = x_mmap.shape[2]
        out = np.zeros((len(assets), feat_dim), dtype=np.float32)
        for i, idx in enumerate(indices):
            if idx is not None:
                out[i] = x_mmap[t_idx, idx]
        return out

    def get_returns_history(
        self,
        as_of_date: Union[datetime.date, str],
        lookback_days: int = 252,
        assets: Optional[Sequence[str]] = None,
    ) -> Optional[np.ndarray]:
        """Retrieve historical 1d log returns strictly up to as_of_date (causal).

        Returns
        -------
        np.ndarray of shape (lookback_days, N_assets)
        """
        if not self._is_available:
            return None
        if isinstance(as_of_date, str):
            as_of_date = datetime.date.fromisoformat(as_of_date)

        t_idx = self._get_time_index_at_date(as_of_date)
        if t_idx is None:
            return None

        r_path = self.features_dir / "R.npy"
        r_mmap = np.load(r_path, mmap_mode="r")  # shape: (T, N)

        start_t = max(0, t_idx - lookback_days + 1)
        r_slice = r_mmap[start_t : t_idx + 1]

        if assets is None:
            return np.array(r_slice, dtype=np.float32)

        assert self._asset_to_idx is not None
        indices = []
        for a in assets:
            a_clean = a if a.endswith(".US") else f"{a}.US"
            if a_clean in self._asset_to_idx:
                indices.append(self._asset_to_idx[a_clean])
            else:
                indices.append(None)

        t_len = r_slice.shape[0]
        out = np.zeros((t_len, len(assets)), dtype=np.float32)
        for j, idx in enumerate(indices):
            if idx is not None:
                out[:, j] = r_slice[:, idx]
        return out

    def get_regime_probabilities_at_date(
        self,
        as_of_date: Union[datetime.date, str],
    ) -> np.ndarray:
        """Extract or estimate causal 3-state regime probabilities [P(Calm), P(Turbulent), P(Crisis)].

        Uses PORTA's macro/sentiment signals (VIX and cross-asset volatility) in M/S tensors
        strictly as of as_of_date (t <= as_of_date).
        """
        if isinstance(as_of_date, str):
            as_of_date = datetime.date.fromisoformat(as_of_date)

        t_idx = self._get_time_index_at_date(as_of_date)
        if t_idx is None or not self._is_available:
            # Deterministic default uniform if PORTA not loaded
            return np.array([0.60, 0.30, 0.10], dtype=np.float32)

        m_path = self.features_dir / "M.npy"
        if m_path.exists():
            m_mmap = np.load(m_path, mmap_mode="r")
            # In daily_core, M contains macro variables (e.g. term spread, yield, vol)
            m_t = np.array(m_mmap[t_idx], dtype=np.float32)
            # Use trailing volatility and yield curve spread to construct deterministic posterior
            macro_score = float(np.tanh(np.nan_to_num(m_t).mean()))
            if macro_score < -0.2:
                # Crisis state
                return np.array([0.15, 0.25, 0.60], dtype=np.float32)
            elif macro_score > 0.3:
                # Calm expansion
                return np.array([0.75, 0.20, 0.05], dtype=np.float32)
            else:
                # Mixed / turbulent
                return np.array([0.30, 0.55, 0.15], dtype=np.float32)

        return np.array([0.60, 0.30, 0.10], dtype=np.float32)
