"""Node feature computation — builds the v_i(t) vector for each asset node.

Implements the 18-dim feature vector from DyFO Manual §2.2:
  retorno_log_21d (1) + vol_hist_21d (1) + beta_mercado (1) +
  setor_one_hot (11) + market_cap_norm (1) + drawdown_atual (1) +
  regime_prob (K=3) + volume_norm (1) = 20 dims (K=3)

Note: actual dim depends on num_regimes K.  Default K=3 → 20 dims total.
When RDM is not available, regime_prob is zero-filled.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from dyfo.config import DataConfig

logger = logging.getLogger(__name__)


class NodeFeatureBuilder:
    """Computes time-varying node feature matrices from market data."""

    def __init__(
        self,
        tickers: List[str],
        ticker_to_idx: Dict[str, int],
        gics_sectors: List[str],
        num_regimes: int = 3,
    ):
        self._tickers = tickers
        self._ticker_to_idx = ticker_to_idx
        self._gics_sectors = gics_sectors
        self._num_regimes = num_regimes
        self._sector_map: Dict[str, int] = {s: i for i, s in enumerate(gics_sectors)}

    @property
    def feature_dim(self) -> int:
        # ret(1) + vol(1) + beta(1) + sector(11) + mcap(1) + dd(1) + regime(K) + vol_norm(1)
        return 1 + 1 + 1 + len(self._gics_sectors) + 1 + 1 + self._num_regimes + 1

    def build_static_features(
        self,
        ticker_info: Dict[str, dict],
    ) -> torch.Tensor:
        """Build the static portion of node features (sector one-hot, log market cap).

        Returns tensor of shape (num_nodes, 11 + 1) = (N, 12).
        """
        n = len(self._tickers)
        num_sectors = len(self._gics_sectors)
        static = torch.zeros(n, num_sectors + 1)

        for ticker in self._tickers:
            idx = self._ticker_to_idx[ticker]
            info = ticker_info.get(ticker, {})
            # Sector one-hot
            sector = info.get("sector", "Unknown")
            s_idx = self._sector_map.get(sector, -1)
            if 0 <= s_idx < num_sectors:
                static[idx, s_idx] = 1.0
            # Log market cap (normalised later)
            mcap = info.get("market_cap")
            if mcap and mcap > 0:
                static[idx, num_sectors] = np.log(mcap)

        # Normalise log market cap across nodes
        mcap_col = static[:, num_sectors]
        mask = mcap_col > 0
        if mask.any():
            mean_mcap = mcap_col[mask].mean()
            std_mcap = mcap_col[mask].std().clamp(min=1e-8)
            static[:, num_sectors] = torch.where(
                mask, (mcap_col - mean_mcap) / std_mcap, torch.zeros_like(mcap_col)
            )
        return static

    def build_daily_features(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame],
        benchmark_prices: Optional[pd.Series],
        ticker_info: Dict[str, dict],
        regime_probs: Optional[pd.DataFrame] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build the full node feature tensor for each trading day.

        Parameters
        ----------
        prices : DataFrame
            Columns = tickers, rows = dates, values = adjusted close.
        volumes : DataFrame or None
            Same shape as prices.
        benchmark_prices : Series or None
            Benchmark index (e.g. SPY) prices, same date index.
        ticker_info : dict
            Output of yfinance_adapter.get_ticker_info().
        regime_probs : DataFrame or None
            Shape (T, K) with regime probabilities from RDM.
            If None, zero-filled.

        Returns
        -------
        dict mapping date_str -> torch.Tensor of shape (N, feature_dim)
        """
        n = len(self._tickers)
        static = self.build_static_features(ticker_info)

        log_ret = np.log(prices / prices.shift(1))
        ret_21d = log_ret.rolling(21).sum()
        vol_21d = log_ret.rolling(21).std()

        # Beta vs benchmark (rolling 63d)
        beta_df = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
        if benchmark_prices is not None:
            bench_ret = np.log(benchmark_prices / benchmark_prices.shift(1))
            for ticker in prices.columns:
                cov = log_ret[ticker].rolling(63).cov(bench_ret)
                var_bench = bench_ret.rolling(63).var()
                beta_df[ticker] = cov / var_bench.replace(0, np.nan)

        # Drawdown
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax.replace(0, np.nan)

        # Normalised volume
        if volumes is not None:
            vol_mean = volumes.rolling(21).mean()
            vol_norm = volumes / vol_mean.replace(0, np.nan)
        else:
            vol_norm = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        features_by_date: Dict[str, torch.Tensor] = {}

        for date in prices.index:
            node_feat = torch.zeros(n, self.feature_dim)

            for ticker in prices.columns:
                idx = self._ticker_to_idx.get(ticker)
                if idx is None:
                    continue

                col = 0
                # retorno_log_21d
                v = ret_21d.at[date, ticker]
                node_feat[idx, col] = 0.0 if pd.isna(v) else float(v)
                col += 1

                # vol_hist_21d
                v = vol_21d.at[date, ticker]
                node_feat[idx, col] = 0.0 if pd.isna(v) else float(v)
                col += 1

                # beta_mercado
                v = beta_df.at[date, ticker]
                node_feat[idx, col] = 0.0 if pd.isna(v) else float(v)
                col += 1

                # setor_one_hot (11) + market_cap_norm (1) -- from static
                node_feat[idx, col : col + len(self._gics_sectors) + 1] = static[idx]
                col += len(self._gics_sectors) + 1

                # drawdown_atual
                v = drawdown.at[date, ticker]
                node_feat[idx, col] = 0.0 if pd.isna(v) else float(v)
                col += 1

                # regime_prob (K dims)
                if regime_probs is not None and date in regime_probs.index:
                    node_feat[idx, col : col + self._num_regimes] = torch.tensor(
                        regime_probs.loc[date].values[: self._num_regimes],
                        dtype=torch.float32,
                    )
                col += self._num_regimes

                # volume_norm
                v = vol_norm.at[date, ticker]
                node_feat[idx, col] = 0.0 if pd.isna(v) else float(v)

            features_by_date[str(date.date())] = node_feat

        logger.info(
            "Built node features for %d dates, %d nodes, dim=%d",
            len(features_by_date),
            n,
            self.feature_dim,
        )
        return features_by_date
