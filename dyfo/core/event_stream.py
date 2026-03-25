"""Financial event definitions and event stream builder.

Converts raw data from yfinance/FRED into a unified temporal event stream
consumed by the TGN encoder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event type enum
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    PRICE_UPDATE = "PRICE_UPDATE"
    EARNINGS_REPORT = "EARNINGS_REPORT"
    FED_DECISION = "FED_DECISION"
    CREDIT_DOWNGRADE = "CREDIT_DOWNGRADE"
    CORP_ACTION = "CORP_ACTION"
    CORRELATION_UPDATE = "CORRELATION_UPDATE"
    MACRO_RELEASE = "MACRO_RELEASE"

    @property
    def feature_dim(self) -> int:
        """Expected raw feature dimension per event type."""
        return _EVENT_FEATURE_DIMS[self]


_EVENT_FEATURE_DIMS: Dict[EventType, int] = {
    EventType.PRICE_UPDATE: 3,        # [delta_ret, vol_1d, volume_norm]
    EventType.EARNINGS_REPORT: 3,     # [surprise_eps, revenue_beat, guidance_delta]
    EventType.FED_DECISION: 3,        # [delta_rate, dot_plot_revision, sentiment]
    EventType.CREDIT_DOWNGRADE: 3,    # [notch_delta, outlook_code, sector_contagion]
    EventType.CORP_ACTION: 3,         # [event_type_code, deal_value_norm, premium]
    EventType.CORRELATION_UPDATE: 3,  # [rho_new, delta_rho, significance]
    EventType.MACRO_RELEASE: 3,       # [surprise_z, revision, vol_impact]
}


# ---------------------------------------------------------------------------
# Financial Event dataclass
# ---------------------------------------------------------------------------

@dataclass
class FinancialEvent:
    """A single temporal event in the DyFO event stream."""

    event_type: EventType
    timestamp: float          # fractional days since epoch (for continuous time)
    source_node: int          # index of asset i
    target_node: int          # index of asset j (-1 for single-node events)
    edge_type: Optional[str]  # 'CORR', 'SECT', etc; None for node-level events
    features: torch.Tensor    # shape (feature_dim,)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "edge_type": self.edge_type,
            "features": self.features.tolist(),
        }


# ---------------------------------------------------------------------------
# Epoch conversion
# ---------------------------------------------------------------------------

_EPOCH = pd.Timestamp("2000-01-01")


def timestamp_to_float(ts: pd.Timestamp) -> float:
    """Convert timestamp to fractional days since epoch."""
    return (ts - _EPOCH).total_seconds() / 86400.0


# ---------------------------------------------------------------------------
# Event stream builder
# ---------------------------------------------------------------------------

class EventStreamBuilder:
    """Builds a sorted list of FinancialEvents from raw data sources."""

    def __init__(self, ticker_to_idx: Dict[str, int]):
        self._ticker_to_idx = ticker_to_idx

    # ---- PRICE_UPDATE ----
    def build_price_events(
        self,
        prices: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> List[FinancialEvent]:
        """Create PRICE_UPDATE events from daily close prices.

        Parameters
        ----------
        prices : DataFrame
            Columns = tickers, index = DatetimeIndex.
        volumes : DataFrame, optional
            Same shape; used to compute normalised volume.
        """
        log_ret = np.log(prices / prices.shift(1))
        vol_1d = log_ret.rolling(21).std()

        if volumes is not None:
            vol_mean_21d = volumes.rolling(21).mean()
            vol_norm = volumes / vol_mean_21d.replace(0, np.nan)
        else:
            vol_norm = pd.DataFrame(
                np.zeros_like(prices.values),
                index=prices.index,
                columns=prices.columns,
            )

        events: List[FinancialEvent] = []
        for date in log_ret.index[1:]:  # skip first row (NaN returns)
            ts = timestamp_to_float(pd.Timestamp(date))
            for ticker in prices.columns:
                idx = self._ticker_to_idx.get(ticker)
                if idx is None:
                    continue
                ret = log_ret.at[date, ticker]
                vol = vol_1d.at[date, ticker]
                vnorm = vol_norm.at[date, ticker]
                if pd.isna(ret):
                    continue
                feat = torch.tensor(
                    [
                        ret if not pd.isna(ret) else 0.0,
                        vol if not pd.isna(vol) else 0.0,
                        vnorm if not pd.isna(vnorm) else 0.0,
                    ],
                    dtype=torch.float32,
                )
                events.append(
                    FinancialEvent(
                        event_type=EventType.PRICE_UPDATE,
                        timestamp=ts,
                        source_node=idx,
                        target_node=-1,
                        edge_type=None,
                        features=feat,
                    )
                )
        logger.info("Built %d PRICE_UPDATE events", len(events))
        return events

    # ---- EARNINGS_REPORT ----
    def build_earnings_events(
        self,
        earnings_df: pd.DataFrame,
    ) -> List[FinancialEvent]:
        """Create EARNINGS_REPORT events from earnings data.

        Parameters
        ----------
        earnings_df : DataFrame
            Columns: ticker, date, eps_estimate, eps_actual, surprise.
        """
        events: List[FinancialEvent] = []
        for _, row in earnings_df.iterrows():
            idx = self._ticker_to_idx.get(row["ticker"])
            if idx is None:
                continue
            surprise = row.get("surprise", 0.0)
            eps_est = row.get("eps_estimate", 0.0)
            eps_act = row.get("eps_actual", 0.0)
            feat = torch.tensor(
                [
                    float(surprise) if not pd.isna(surprise) else 0.0,
                    float(eps_act - eps_est) if not (pd.isna(eps_act) or pd.isna(eps_est)) else 0.0,
                    0.0,  # guidance_delta — not available from yfinance
                ],
                dtype=torch.float32,
            )
            events.append(
                FinancialEvent(
                    event_type=EventType.EARNINGS_REPORT,
                    timestamp=timestamp_to_float(pd.Timestamp(row["date"])),
                    source_node=idx,
                    target_node=-1,
                    edge_type=None,
                    features=feat,
                )
            )
        logger.info("Built %d EARNINGS_REPORT events", len(events))
        return events

    # ---- CORP_ACTION ----
    def build_corp_action_events(
        self,
        actions_df: pd.DataFrame,
    ) -> List[FinancialEvent]:
        """Create CORP_ACTION events from splits/dividends data.

        Parameters
        ----------
        actions_df : DataFrame
            Columns: ticker, date, action_type, value.
        """
        action_codes = {"SPLIT": 1.0, "DIVIDEND": 2.0}
        events: List[FinancialEvent] = []
        for _, row in actions_df.iterrows():
            idx = self._ticker_to_idx.get(row["ticker"])
            if idx is None:
                continue
            code = action_codes.get(row["action_type"], 0.0)
            feat = torch.tensor(
                [code, float(row["value"]), 0.0],
                dtype=torch.float32,
            )
            events.append(
                FinancialEvent(
                    event_type=EventType.CORP_ACTION,
                    timestamp=timestamp_to_float(pd.Timestamp(row["date"])),
                    source_node=idx,
                    target_node=-1,
                    edge_type=None,
                    features=feat,
                )
            )
        logger.info("Built %d CORP_ACTION events", len(events))
        return events

    # ---- MACRO_RELEASE / FED_DECISION ----
    def build_macro_events(
        self,
        macro_events_df: pd.DataFrame,
        num_nodes: int,
    ) -> List[FinancialEvent]:
        """Create MACRO_RELEASE and FED_DECISION events.

        FED_DECISION events affect ALL nodes simultaneously.
        Other macro releases are broadcast to all nodes as MACRO_RELEASE.

        Parameters
        ----------
        macro_events_df : DataFrame
            Columns: date, series, value, change, surprise_z.
        num_nodes : int
            Total number of asset nodes.
        """
        events: List[FinancialEvent] = []
        for _, row in macro_events_df.iterrows():
            ts = timestamp_to_float(pd.Timestamp(row["date"]))
            is_fed = row["series"] == "fed_funds_rate"
            etype = EventType.FED_DECISION if is_fed else EventType.MACRO_RELEASE
            feat = torch.tensor(
                [
                    float(row.get("surprise_z", 0.0)),
                    float(row.get("change", 0.0)),
                    0.0,  # vol_impact placeholder
                ],
                dtype=torch.float32,
            )
            # Broadcast to all nodes
            for node_idx in range(num_nodes):
                events.append(
                    FinancialEvent(
                        event_type=etype,
                        timestamp=ts,
                        source_node=node_idx,
                        target_node=-1,
                        edge_type=None,
                        features=feat,
                    )
                )
        logger.info(
            "Built %d macro/fed events (broadcast to %d nodes each)",
            len(events),
            num_nodes,
        )
        return events

    # ---- CORRELATION_UPDATE ----
    def build_correlation_events(
        self,
        corr_series: pd.DataFrame,
        pairs: List[tuple],
    ) -> List[FinancialEvent]:
        """Create CORRELATION_UPDATE events from time-varying correlations.

        Parameters
        ----------
        corr_series : DataFrame
            Index = dates, columns = pair identifiers (e.g. "AAPL_MSFT"),
            values = rho(t).
        pairs : list of (ticker_i, ticker_j) tuples
            Corresponding to columns of corr_series.
        """
        events: List[FinancialEvent] = []
        prev_rho: Dict[str, float] = {}

        for date in corr_series.index:
            ts = timestamp_to_float(pd.Timestamp(date))
            for pair_col, (tk_i, tk_j) in zip(corr_series.columns, pairs):
                idx_i = self._ticker_to_idx.get(tk_i)
                idx_j = self._ticker_to_idx.get(tk_j)
                if idx_i is None or idx_j is None:
                    continue
                rho = corr_series.at[date, pair_col]
                if pd.isna(rho):
                    continue
                delta_rho = rho - prev_rho.get(pair_col, rho)
                prev_rho[pair_col] = rho
                feat = torch.tensor(
                    [rho, delta_rho, 1.0],  # significance=1.0 placeholder
                    dtype=torch.float32,
                )
                events.append(
                    FinancialEvent(
                        event_type=EventType.CORRELATION_UPDATE,
                        timestamp=ts,
                        source_node=idx_i,
                        target_node=idx_j,
                        edge_type="CORR",
                        features=feat,
                    )
                )
        logger.info("Built %d CORRELATION_UPDATE events", len(events))
        return events

    # ---- Merge & sort ----
    @staticmethod
    def merge_and_sort(*event_lists: List[FinancialEvent]) -> List[FinancialEvent]:
        """Merge multiple event lists and sort by timestamp."""
        merged: List[FinancialEvent] = []
        for lst in event_lists:
            merged.extend(lst)
        merged.sort(key=lambda e: e.timestamp)
        return merged
