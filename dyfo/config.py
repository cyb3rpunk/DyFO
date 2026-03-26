"""DyFO configuration dataclasses."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DyFOConfig:
    """Master configuration for the DyFO module."""

    # --- Graph dimensions ---
    memory_dim: int = 172
    embedding_dim: int = 100
    time_encoding_dim: int = 100
    edge_type_embedding_dim: int = 16
    num_attention_heads: int = 2
    num_neighbors: int = 10
    num_gat_layers: int = 1

    # --- Edge types ---
    edge_types: List[str] = field(
        default_factory=lambda: ["CORR", "SECT", "SUPL", "FACT"]
    )

    # --- Event types ---
    event_types: List[str] = field(
        default_factory=lambda: [
            "PRICE_UPDATE",
            "EARNINGS_REPORT",
            "FED_DECISION",
            "CREDIT_DOWNGRADE",
            "CORP_ACTION",
            "CORRELATION_UPDATE",
            "MACRO_RELEASE",
        ]
    )

    # --- Training ---
    dropout: float = 0.1
    batch_size_events: int = 200
    staleness_threshold_days: int = 5
    corr_sparsify_threshold: float = 0.3

    # --- Correlation ---
    correlation_method: str = "dcc_garch"  # "rolling_pearson" or "dcc_garch"
    dcc_garch_window: int = 252
    rolling_corr_window: int = 63

    # --- Data ---
    node_feature_dim: int = 20  # 1+1+1+11+1+1+K(3)+1 = 20 (see manual §2.2 with K=3)
    num_regimes: int = 3  # K regimes from RDM


@dataclass
class DataConfig:
    """Configuration for data acquisition from free APIs."""

    # --- Universe ---
    tickers: List[str] = field(default_factory=list)
    benchmark_ticker: str = "SPY"
    start_date: str = "2018-01-01"
    end_date: str = "2025-12-31"

    # --- FRED series IDs ---
    fred_api_key: str = ""
    fred_series: dict = field(
        default_factory=lambda: {
            "fed_funds_rate": "DFF",
            "vix": "VIXCLS",
            "credit_spread": "BAMLC0A0CM",  # ICE BofA US Corp Master OAS
            "yield_10y": "DGS10",
            "yield_2y": "DGS2",
            "cpi_yoy": "CPIAUCSL",
            "unemployment": "UNRATE",
            "pmi_manufacturing": "MANEMP",
        }
    )

    # --- Sector mapping (GICS-like, 11 sectors) ---
    gics_sectors: List[str] = field(
        default_factory=lambda: [
            "Energy",
            "Materials",
            "Industrials",
            "Consumer Discretionary",
            "Consumer Staples",
            "Health Care",
            "Financials",
            "Information Technology",
            "Communication Services",
            "Utilities",
            "Real Estate",
        ]
    )
