"""Common interface for DyFO graph encoder variants (BL-02).

All encoder variants expose four methods:

  reset_state()            — zero temporal state (call at epoch start)
  advance_day(...)         — process one trading day, update state (no grad)
  get_node_embeddings(...) — compute (num_nodes, embedding_dim) embeddings
  detach_state()           — detach buffers from computation graph after
                             backward() (TBPTT); no-op for stateless variants

Use ``build_encoder(config, num_nodes)`` to instantiate the right variant.

Variants
--------
"tgn"        — original DyFO Temporal Graph Network (DyFOModule wrapper)
"gat_static" — 2-layer GAT on a static mean-correlation graph (GATStaticEncoder)
"roland"     — ROLAND-like monthly snapshot GNN with EMA state update
               (ROLANDLikeEncoder; see roland_baseline.py for caveats)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import torch
import torch.nn as nn

from dyfo.config import DyFOConfig
from dyfo.core.event_stream import FinancialEvent


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseGraphEncoder(nn.Module, ABC):
    """Abstract base class for all DyFO encoder variants.

    Subclass this and implement the four abstract methods to add a new variant.
    The decoder (``LinkPredictor`` / ``CorrelationRegressor``) and the
    walk-forward training loop are variant-agnostic.
    """

    def __init__(self, config: DyFOConfig, num_nodes: int):
        super().__init__()
        self.config = config
        self.num_nodes = num_nodes
        self.embedding_dim = config.embedding_dim

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @abstractmethod
    def reset_state(self) -> None:
        """Zero all temporal state (memory buffers, cached embeddings).

        Call once at the start of each training epoch before iterating
        over trading days.
        """

    @abstractmethod
    def advance_day(
        self,
        events: List[FinancialEvent],
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_timestamps: torch.Tensor,
        current_time: float,
    ) -> None:
        """Update internal temporal state for one trading day.

        This call should NOT require gradients — all persistent state
        should be stored as detached tensors (TBPTT).

        Parameters
        ----------
        events
            Events observed on today (may be empty).
        node_features
            Tensor (num_nodes, node_feature_dim) for today.
        edge_index
            Static graph edge index (2, num_edges).
        edge_type_ids
            Integer edge-type IDs (num_edges,).
        edge_timestamps
            Float edge timestamps (num_edges,).
        current_time
            Today's timestamp as a float (fractional days since epoch).
        """

    @abstractmethod
    def get_node_embeddings(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_timestamps: torch.Tensor,
        current_time: float,
    ) -> torch.Tensor:
        """Compute node embeddings from current state (differentiable).

        ``loss.backward()`` will reach encoder parameters through the
        returned tensor.

        Returns
        -------
        Tensor of shape (num_nodes, embedding_dim).
        """

    # ------------------------------------------------------------------
    # Optional hook — override only when temporal state needs detachment
    # ------------------------------------------------------------------

    def detach_state(self) -> None:
        """Detach temporal state from the computation graph.

        Call this *after* ``loss.backward()`` (and before the next day's
        ``advance_day``) to implement truncated BPTT.  The default is a
        no-op; override in variants that carry a live memory tensor.
        """


# ---------------------------------------------------------------------------
# TGN wrapper
# ---------------------------------------------------------------------------

class TGNWrapper(BaseGraphEncoder):
    """Wraps ``DyFOModule`` to conform to ``BaseGraphEncoder``.

    The underlying ``DyFOModule`` + ``TGNEncoder`` are fully preserved;
    this class is a thin adapter that re-routes calls through the common
    interface.

    The ``encoder`` attribute is exposed so that existing code that reads
    ``wrapper.encoder.memory`` keeps working.
    """

    def __init__(self, config: DyFOConfig, num_nodes: int):
        super().__init__(config, num_nodes)
        from dyfo.core.dyfo_module import DyFOModule

        self._module = DyFOModule(
            config=config, num_nodes=num_nodes, readout_strategy="mean"
        )
        # Forward attribute access so callers can still do encoder.encoder.memory
        self.encoder = self._module.encoder

    # ------------------------------------------------------------------
    # BaseGraphEncoder interface
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        self._module.reset_memory()

    def advance_day(
        self,
        events: List[FinancialEvent],
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_timestamps: torch.Tensor,
        current_time: float,
    ) -> None:
        """Process today's events and update TGN memory (no gradient)."""
        self._module.process_day_events(events)

    def get_node_embeddings(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        edge_timestamps: torch.Tensor,
        current_time: float,
    ) -> torch.Tensor:
        """Run GAT attention over the graph and return (N, embedding_dim)."""
        return self._module.encoder.compute_embeddings(
            node_features=node_features,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            edge_timestamps=edge_timestamps,
            current_time=current_time,
        )

    def detach_state(self) -> None:
        """Detach TGN memory from computation graph (TBPTT)."""
        self._module.encoder.memory = self._module.encoder.memory.detach()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_encoder(
    config: DyFOConfig,
    num_nodes: int,
    variant: Optional[str] = None,
    **kwargs,
) -> BaseGraphEncoder:
    """Instantiate the encoder variant named by ``config.model_variant``.

    Parameters
    ----------
    config
        DyFO configuration object.
    num_nodes
        Number of asset nodes in the universe.
    variant
        Override ``config.model_variant``.  Useful for tests.
    **kwargs
        Passed through to the variant constructor (e.g.
        ``corr_labels_by_date`` for ``ROLANDLikeEncoder``).

    Returns
    -------
    BaseGraphEncoder subclass (also an nn.Module).
    """
    v = variant if variant is not None else config.model_variant

    if v == "tgn":
        return TGNWrapper(config, num_nodes)

    if v == "gat_static":
        from dyfo.core.gat_static_baseline import GATStaticEncoder

        return GATStaticEncoder(config, num_nodes)

    if v == "roland":
        from dyfo.core.roland_baseline import ROLANDLikeEncoder

        return ROLANDLikeEncoder(config, num_nodes, **kwargs)

    raise ValueError(
        f"Unknown model_variant {v!r}. "
        "Valid choices: 'tgn', 'gat_static', 'roland'."
    )
