"""ROLAND-like baseline (BL-02) — snapshot GNN with temporal state evolution.

IMPORTANT — nomenclature
-------------------------
This is a *ROLAND-like* model, NOT a faithful re-implementation of
You et al. (2022) "ROLAND: Graph Neural Networks for Dynamic Graphs"
(https://arxiv.org/abs/2208.07239).

Key differences from the original ROLAND:
  1. Financial correlation matrices are used as snapshots (monthly frequency),
     not arbitrary timestamped edge streams.
  2. State update is EMA — no GRU-based hierarchical update as in ROLAND.
  3. No multi-level snapshot aggregation from the original paper.
  4. Architecture is 2-layer multi-head GAT (not GraphSAGE as in ROLAND).

What this model *does* share with ROLAND's spirit:
  - Discrete temporal snapshots (monthly correlation matrices).
  - Temporal state carried across snapshots without continuous-time memory.
  - GNN runs on each snapshot; output blended with previous state via EMA.

Architecture
------------
For each monthly snapshot G_m:
  h_gat  = GAT(G_m, x_today)                  (N, embedding_dim)
  h_new  = (1 - α) * h_prev.detach() + α * h_gat   EMA blend

  h_prev is updated to h_new.detach() after each call to get_node_embeddings.

Usage
-----
  encoder = ROLANDLikeEncoder(config, num_nodes)
  encoder.precompute_monthly_snapshots(corr_labels_by_date)
  # ... BaseGraphEncoder interface from here ...
"""

from __future__ import annotations

import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from dyfo.config import DyFOConfig
from dyfo.core.event_stream import FinancialEvent
from dyfo.core.gat_static_baseline import MultiHeadGATLayer
from dyfo.core.model_variants import BaseGraphEncoder

# Epoch used by timestamp_to_float (must match event_stream.py)
_EPOCH = datetime.date(2000, 1, 1)


def _date_key_to_year_month(date_key: int) -> Tuple[int, int]:
    """Convert an integer date key (days since 2000-01-01) to (year, month).

    ``date_key`` is obtained via ``int(timestamp_to_float(pd.Timestamp(date)))``,
    i.e. the whole number of days elapsed since 2000-01-01.
    """
    dt = _EPOCH + datetime.timedelta(days=date_key)
    return (dt.year, dt.month)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class ROLANDLikeEncoder(BaseGraphEncoder):
    """ROLAND-like encoder with monthly correlation snapshots.

    The active snapshot graph changes on the first trading day of each new
    calendar month (detected in ``advance_day``).  Within a month the graph
    structure is constant; node features are refreshed every day.

    State update per day
    --------------------
    Let G_m be the monthly snapshot active on day t and x_t be today's node
    features.  Then:

        h_gat  = GAT_2layer(G_m, x_t)
        h_new  = (1 - α) * h_prev.detach() + α * h_gat

    where α = ``config.roland_ema_alpha`` and h_prev starts at zeros.

    ``h_new.detach()`` is stored as ``_h_prev`` so gradients do not
    accumulate across time steps (TBPTT analogue).

    Parameters
    ----------
    config
        DyFO configuration.
    num_nodes
        Number of asset nodes.
    corr_labels_by_date
        Optional pre-loaded mapping date_key → {(i, j): rho}.  If provided,
        ``precompute_monthly_snapshots`` is called automatically.
    """

    def __init__(
        self,
        config: DyFOConfig,
        num_nodes: int,
        corr_labels_by_date: Optional[Dict[int, Dict[Tuple[int, int], float]]] = None,
    ):
        super().__init__(config, num_nodes)

        node_feat_dim = config.node_feature_dim
        num_heads = config.num_attention_heads
        hidden_per_head = max(32, config.embedding_dim // num_heads)
        self.ema_alpha: float = getattr(config, "roland_ema_alpha", 0.5)

        # 2-layer multi-head GAT (same structure as GATStaticEncoder)
        self.gat1 = MultiHeadGATLayer(
            in_dim=node_feat_dim,
            out_dim=hidden_per_head,
            num_heads=num_heads,
            dropout=config.dropout,
            concat=True,
        )
        self.gat2 = MultiHeadGATLayer(
            in_dim=hidden_per_head * num_heads,
            out_dim=config.embedding_dim,
            num_heads=num_heads,
            dropout=config.dropout,
            concat=False,
        )

        # Node hidden state from the previous snapshot (always detached)
        self.register_buffer(
            "_h_prev", torch.zeros(num_nodes, config.embedding_dim)
        )
        # Active snapshot edge index
        self.register_buffer(
            "_snap_edge_index", torch.zeros(2, 0, dtype=torch.long)
        )

        # Current month as (year, month); None = not yet set
        self._current_month: Optional[Tuple[int, int]] = None

        # Per-month pre-computed edge indices: (year, month) -> (2, E) tensor
        self._monthly_snapshots: Dict[Tuple[int, int], torch.Tensor] = {}

        if corr_labels_by_date is not None:
            self.precompute_monthly_snapshots(corr_labels_by_date)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def precompute_monthly_snapshots(
        self,
        corr_labels_by_date: Dict[int, Dict[Tuple[int, int], float]],
        threshold: Optional[float] = None,
    ) -> None:
        """Build a per-month sparse edge index from daily correlation data.

        For each calendar month all available days' |rho| values are averaged;
        pairs whose mean |rho| meets ``threshold`` become edges.

        Parameters
        ----------
        corr_labels_by_date
            Mapping date_key → {(i, j): rho}.
        threshold
            Minimum mean |rho| for an edge.  Defaults to
            ``config.corr_sparsify_threshold``.
        """
        if threshold is None:
            threshold = self.config.corr_sparsify_threshold

        monthly_sums: Dict[Tuple[int, int], Dict[Tuple[int, int], float]] = {}
        monthly_counts: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}

        for date_key, corr_dict in corr_labels_by_date.items():
            ym = _date_key_to_year_month(int(date_key))
            sums = monthly_sums.setdefault(ym, {})
            counts = monthly_counts.setdefault(ym, {})
            for (i, j), rho in corr_dict.items():
                pair = (min(i, j), max(i, j))
                sums[pair] = sums.get(pair, 0.0) + abs(rho)
                counts[pair] = counts.get(pair, 0) + 1

        self._monthly_snapshots = {}
        for ym, sums in monthly_sums.items():
            src_list: List[int] = []
            dst_list: List[int] = []
            for (i, j), total in sums.items():
                mean_abs_rho = total / monthly_counts[ym][(i, j)]
                if mean_abs_rho >= threshold:
                    src_list += [i, j]
                    dst_list += [j, i]
            if src_list:
                ei = torch.tensor([src_list, dst_list], dtype=torch.long)
            else:
                ei = torch.zeros(2, 0, dtype=torch.long)
            self._monthly_snapshots[ym] = ei

    # ------------------------------------------------------------------
    # BaseGraphEncoder interface
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Zero the node hidden state and reset the active snapshot."""
        self._h_prev.zero_()
        self._snap_edge_index = torch.zeros(2, 0, dtype=torch.long)
        self._current_month = None

    def advance_day(
        self,
        events: List[FinancialEvent],
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_timestamps: torch.Tensor,
        current_time: float,
    ) -> None:
        """Detect month boundaries and switch the active snapshot.

        No gradient computation occurs here.  The EMA update and state
        write-back happen inside ``get_node_embeddings`` so that GNN
        parameters receive gradients from the decoder loss.
        """
        ym = _date_key_to_year_month(int(current_time))
        if ym != self._current_month:
            self._current_month = ym
            snap = self._monthly_snapshots.get(ym)
            if snap is not None:
                self._snap_edge_index = snap
            # If no snapshot exists for this month (data gap), the previous
            # snapshot remains active as a reasonable fallback.

    def get_node_embeddings(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_timestamps: torch.Tensor,
        current_time: float,
    ) -> torch.Tensor:
        """Compute EMA-blended GAT embeddings for the current snapshot.

        EMA blend:
            h_new = (1 - α) * h_prev.detach() + α * GAT(G_month, x_today)

        State ``_h_prev`` is overwritten with ``h_new.detach()`` so the
        next call starts from the current output.

        The ``edge_index`` argument is ignored; the active monthly snapshot
        is used instead.

        Returns
        -------
        Tensor of shape (num_nodes, embedding_dim) — differentiable w.r.t.
        GAT parameters.
        """
        ei = self._snap_edge_index.to(node_features.device)

        # GAT forward (differentiable)
        h_gat = self.gat1(node_features, ei)
        h_gat = self.gat2(h_gat, ei)  # (N, embedding_dim)

        # EMA blend — h_prev is always detached so BPTT stops here
        h_prev = self._h_prev.detach()
        h_new = (1.0 - self.ema_alpha) * h_prev + self.ema_alpha * h_gat

        # Write back detached state for the next call
        self._h_prev = h_new.detach()

        return h_new
