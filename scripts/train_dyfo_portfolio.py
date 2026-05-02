#!/usr/bin/env python3
"""Train DyFO (TGAT) on a multi-asset universe and run DRL portfolio episodes.

Universe
--------
  Stocks  : 15 S&P 500 stocks covering all 11 GICS sectors
  Bonds   : TLT  (iShares 20+ Year Treasury)
  Gold    : GLD  (SPDR Gold Shares)
  Crypto  : BTC-USD (Bitcoin)

Two-phase workflow
------------------
  Phase 1 -- train:
    Download market data, compute rolling correlations, train TGAT via
    self-supervised link prediction (regression mode), save checkpoint to disk.

  Phase 2 -- drl:
    Load saved checkpoint, run multiple DRL episodes over the test window.
    Three conditions are evaluated on the same episode data:

      [A] DyFO-DRL  -- state = per-asset TGAT embeddings Z in R^(N x d)
                       Policy: asset-wise MLP -> softmax weights.
      [B] Raw-DRL   -- state = raw price features in R^(N x 3)
                       Ablation: same policy without graph structure.
      [C] EWMA-GMVP -- no learning; closed-form min-variance portfolio
                       from EWMA covariance (alpha=0.05).

Usage
-----
  # Full run (train + DRL):
  python scripts/train_dyfo_portfolio.py --phase all --epochs 15 --drl_episodes 30

  # Train only, save checkpoint:
  python scripts/train_dyfo_portfolio.py --phase train --epochs 15

  # DRL only (load existing checkpoint):
  python scripts/train_dyfo_portfolio.py --phase drl --checkpoint results/dyfo_portfolio_ckpt.pt

Output
------
  results/dyfo_portfolio_ckpt.pt       -- TGAT encoder + decoder weights + metadata
  results/dyfo_drl_policy.pt           -- Trained DRL policy weights
  results/dyfo_portfolio_report.json   -- Per-episode metrics for all 3 conditions
"""

from __future__ import annotations

import argparse
import bisect
import datetime
import json
import math
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dyfo.config import DataConfig, DyFOConfig
from dyfo.core.link_prediction import (
    CorrelationRegressor,
    build_regression_labels,
    compute_regression_metrics,
)
from dyfo.core.model_variants import build_encoder
from dyfo.logging_utils import RESULTS_DIR, setup_logging
from scripts.run_bootstrap_eval_v5 import TGN_LR, TGN_PATIENCE, load_or_prepare_data
from scripts.train_link_prediction import set_seed

# ─────────────────────────────────────────────────────────────────────────────
# Universe
# ─────────────────────────────────────────────────────────────────────────────

# 15 S&P 500 stocks -- all 11 GICS sectors represented
STOCKS: List[str] = [
    "AAPL", "MSFT",   # Information Technology
    "JPM", "GS",      # Financials
    "JNJ", "UNH",     # Health Care
    "AMZN", "TSLA",   # Consumer Discretionary
    "PG",             # Consumer Staples
    "XOM",            # Energy
    "CAT",            # Industrials
    "META",           # Communication Services
    "LIN",            # Materials
    "NEE",            # Utilities
    "PLD",            # Real Estate
]

ALTS: List[str] = [
    "TLT",      # iShares 20+ Year Treasury Bond ETF (rates / duration risk)
    "GLD",      # SPDR Gold Shares (inflation / safe-haven)
    "BTC-USD",  # Bitcoin (crypto / tail risk / liquidity)
]

UNIVERSE: List[str] = STOCKS + ALTS   # 18 assets total

# ─────────────────────────────────────────────────────────────────────────────
# Split constants
# ─────────────────────────────────────────────────────────────────────────────

DATA_START = "2015-01-01"   # start of history (includes 2015 China crash, 2018 Q4)
DATA_END   = "2024-12-31"
TRAIN_END  = "2022-06-30"   # ~7 years of training, includes COVID and 2022 rate shock
VAL_END    = "2023-06-30"   # 1-year validation
TEST_START = "2023-07-01"   # 1.5-year test window (out-of-sample)
TEST_END   = "2024-12-31"

ALPHA_EWMA = 0.05           # matches project convention
EPISODE_LEN = 60            # trading days per DRL episode (~3 months)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_EPOCH = datetime.date(2000, 1, 1)


def _int_day_to_iso(day: int) -> str:
    return (_EPOCH + datetime.timedelta(days=int(day))).isoformat()


def _slice_dates(sorted_dates: List[int], start: str, end: str) -> List[int]:
    start_ts, end_ts = torch.tensor(0), torch.tensor(0)  # dummy init
    start_ts = datetime.date.fromisoformat(start)
    end_ts   = datetime.date.fromisoformat(end)
    return [
        d for d in sorted_dates
        if start_ts <= datetime.date.fromisoformat(_int_day_to_iso(d)) <= end_ts
    ]


def _node_feature_getter(data: dict):
    """Return a function that looks up the closest node feature tensor for a date."""
    nf_keys = sorted(data["node_features_by_date"].keys())

    def _get(date_key: int) -> torch.Tensor:
        iso = _int_day_to_iso(date_key)
        idx = bisect.bisect_right(nf_keys, iso) - 1
        idx = max(0, idx)
        return data["node_features_by_date"][nf_keys[idx]]

    return _get


def _build_cosine_scheduler(optimizer, num_epochs: int):
    warmup = min(2, num_epochs)

    def _lr(ep: int) -> float:
        if ep < warmup:
            return (ep + 1) / max(1, warmup)
        progress = (ep - warmup) / max(1, num_epochs - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: DyFO training
# ─────────────────────────────────────────────────────────────────────────────

def train_dyfo(
    data: dict,
    num_nodes: int,
    train_dates: List[int],
    val_dates: List[int],
    num_epochs: int,
    lr: float,
    patience: int,
    seed: int,
    device: torch.device,
) -> dict:
    """Train TGAT encoder + correlation decoder; return best state dicts.

    Returns a dict with keys:
      encoder_state, decoder_state, best_val_r2, best_epoch, train_metrics
    """
    set_seed(seed)

    config = DyFOConfig(model_variant="tgat")
    encoder = build_encoder(config, num_nodes, variant="tgat").to(device)
    decoder = CorrelationRegressor(
        embedding_dim=config.embedding_dim,
        hidden_dim=64,
        dropout=config.dropout,
    ).to(device)

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = _build_cosine_scheduler(optimizer, num_epochs)
    loss_fn = nn.SmoothL1Loss()

    graph = data["graph"]
    edge_index    = graph.get_full_edge_index().to(device)
    edge_type_ids = graph.get_edge_type_ids().to(device)
    edge_ts       = torch.zeros(edge_index.shape[1], device=device)
    get_nf        = _node_feature_getter(data)

    def run_split(dates: List[int], train_mode: bool) -> dict:
        if train_mode:
            encoder.train(); decoder.train()
        else:
            encoder.eval(); decoder.eval()

        total: dict = {}
        n = 0
        ctx = torch.enable_grad if train_mode else torch.no_grad

        with ctx():
            for d_idx in range(len(dates) - 1):
                today    = dates[d_idx]
                tomorrow = dates[d_idx + 1]
                events   = data["events_by_date"].get(today, [])
                nf       = get_nf(today).to(device)
                t        = float(today) + 0.99
                labels   = data["corr_labels_by_date"].get(tomorrow, {})

                if not labels:
                    encoder.advance_day(events, nf, edge_index, edge_type_ids, edge_ts, t)
                    continue

                src, dst, targets = build_regression_labels(labels, num_nodes)
                if len(src) == 0:
                    encoder.advance_day(events, nf, edge_index, edge_type_ids, edge_ts, t)
                    continue

                src, dst, targets = src.to(device), dst.to(device), targets.to(device)

                if train_mode:
                    encoder.advance_day(events, nf, edge_index, edge_type_ids, edge_ts, t)
                    z = encoder.get_node_embeddings(nf, edge_index, edge_type_ids, edge_ts, t)
                    preds = decoder(z[src], z[dst])
                    loss  = loss_fn(preds, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        list(encoder.parameters()) + list(decoder.parameters()),
                        max_norm=0.5,
                    )
                    optimizer.step()
                    encoder.detach_state()
                else:
                    encoder.advance_day(events, nf, edge_index, edge_type_ids, edge_ts, t)
                    z = encoder.get_node_embeddings(nf, edge_index, edge_type_ids, edge_ts, t)
                    preds = decoder(z[src], z[dst])

                m = compute_regression_metrics(preds.detach(), targets)
                for k, v in m.items():
                    total[k] = total.get(k, 0.0) + float(v)
                n += 1

        return {k: v / max(1, n) for k, v in total.items()}

    best_val_r2  = float("-inf")
    best_epoch   = 1
    best_state   = None
    wait         = 0

    for epoch in range(1, num_epochs + 1):
        encoder.reset_state()
        tr = run_split(train_dates, train_mode=True)
        vl = run_split(val_dates,   train_mode=False)
        scheduler.step()

        print(
            f"  epoch {epoch:3d}/{num_epochs} | "
            f"train R2={tr.get('r_squared', float('nan')):.4f} "
            f"MAE={tr.get('mae', float('nan')):.4f} | "
            f"val   R2={vl.get('r_squared', float('nan')):.4f} "
            f"MAE={vl.get('mae', float('nan')):.4f}"
        )

        if vl.get("r_squared", float("-inf")) > best_val_r2:
            best_val_r2 = float(vl["r_squared"])
            best_epoch  = epoch
            best_state  = {
                "encoder": {k: v.clone() for k, v in encoder.state_dict().items()},
                "decoder": {k: v.clone() for k, v in decoder.state_dict().items()},
            }
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stop at epoch {epoch} (best epoch {best_epoch})")
                break

    return {
        "encoder_state": best_state["encoder"],
        "decoder_state": best_state["decoder"],
        "best_val_r2":   best_val_r2,
        "best_epoch":    best_epoch,
    }


def save_checkpoint(
    path: Path,
    train_result: dict,
    universe: List[str],
    ticker_to_idx: dict,
    config: DyFOConfig,
) -> None:
    ckpt = {
        "universe":     universe,
        "ticker_to_idx": ticker_to_idx,
        "num_nodes":    len(universe),
        "embedding_dim": config.embedding_dim,
        "model_variant": config.model_variant,
        "encoder_state": train_result["encoder_state"],
        "decoder_state": train_result["decoder_state"],
        "best_val_r2":   train_result["best_val_r2"],
        "best_epoch":    train_result["best_epoch"],
        "data_start":    DATA_START,
        "data_end":      DATA_END,
        "train_end":     TRAIN_END,
        "val_end":       VAL_END,
        "test_start":    TEST_START,
        "test_end":      TEST_END,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    print(f"Checkpoint saved -> {path}")


def load_checkpoint(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(
        f"Checkpoint loaded: {path}  "
        f"(universe={len(ckpt['universe'])} assets, "
        f"best_val_R2={ckpt['best_val_r2']:.4f}, "
        f"epoch={ckpt['best_epoch']})"
    )
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio utilities (shared across all conditions)
# ─────────────────────────────────────────────────────────────────────────────

def _portfolio_log_return(weights: torch.Tensor, next_ret: torch.Tensor) -> torch.Tensor:
    gross = 1.0 + (weights * next_ret).sum()
    return torch.log(gross.clamp(min=1e-8))


def _ewma_cov(history: torch.Tensor, alpha: float) -> torch.Tensor:
    n = history.shape[1]
    cov = torch.eye(n) * 1e-4
    for r in history:
        rv = r.unsqueeze(-1)
        cov = alpha * (rv @ rv.T) + (1.0 - alpha) * cov
    return cov


def _gmvp(cov: torch.Tensor) -> torch.Tensor:
    """Long-only Global Minimum Variance Portfolio weights."""
    n = cov.shape[0]
    reg = cov + 1e-5 * torch.eye(n)
    cov_inv = torch.linalg.inv(reg)
    raw = cov_inv @ torch.ones(n)
    raw = raw.clamp(min=0.0)
    return raw / raw.sum().clamp(min=1e-9)


def _asset_returns_from_prices(prices_df, universe: List[str], device: torch.device) -> torch.Tensor:
    """Daily returns tensor, shape (T, N), aligned to universe order."""
    import pandas as pd
    df = prices_df.reindex(columns=universe).ffill().pct_change().fillna(0.0)
    return torch.tensor(df.values, dtype=torch.float32, device=device)


def _sharpe(log_returns: List[float], annual: int = 252) -> float:
    if len(log_returns) < 2:
        return float("nan")
    r = torch.tensor(log_returns)
    mean = r.mean()
    std  = r.std().clamp(min=1e-9)
    return float(mean / std * math.sqrt(annual))


# ─────────────────────────────────────────────────────────────────────────────
# DRL policy
# ─────────────────────────────────────────────────────────────────────────────

class AssetWisePolicy(nn.Module):
    """Per-asset MLP: applies the same network to each asset's embedding.

    Input  : Z in R^(N, state_dim)  -- per-asset state (embedding or raw features)
    Output : w in Delta^N           -- portfolio weights (simplex via softmax)
    """

    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # Z: (N, state_dim) -> logits: (N,) -> weights: (N,)
        logits = self.net(Z).squeeze(-1)
        return torch.softmax(logits, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Episode result
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeResult(NamedTuple):
    episode_idx: int
    condition: str
    cumulative_log_return: float
    sharpe: float
    mean_entropy: float
    weights_always_valid: bool


def _entropy(w: torch.Tensor) -> float:
    return float(-(w.detach() * (w.detach() + 1e-8).log()).sum())


# ─────────────────────────────────────────────────────────────────────────────
# Condition A: DyFO-DRL
# ─────────────────────────────────────────────────────────────────────────────

def _warm_up_encoder(encoder, data, warm_dates: List[int], edge_index, edge_type_ids, edge_ts, get_nf, device):
    """Advance encoder through warm_dates without gradient (fills event buffers)."""
    encoder.eval()
    with torch.no_grad():
        for d in warm_dates:
            events = data["events_by_date"].get(d, [])
            nf     = get_nf(d).to(device)
            t      = float(d) + 0.99
            encoder.advance_day(events, nf, edge_index, edge_type_ids, edge_ts, t)


def run_dyfo_drl_episodes(
    ckpt: dict,
    data: dict,
    episode_dates_list: List[List[int]],
    warm_dates: List[int],
    config: DyFOConfig,
    device: torch.device,
    n_drl_epochs: int,
    lr_drl: float,
    seed: int,
) -> tuple[List[EpisodeResult], AssetWisePolicy]:
    """Train DRL policy using DyFO embeddings as per-asset state.

    For each DRL epoch: warm up encoder, run one episode, REINFORCE update.
    """
    set_seed(seed)
    num_nodes = ckpt["num_nodes"]

    encoder = build_encoder(config, num_nodes, variant="tgat").to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()   # frozen: only the DRL policy is trained

    graph         = data["graph"]
    edge_index    = graph.get_full_edge_index().to(device)
    edge_type_ids = graph.get_edge_type_ids().to(device)
    edge_ts       = torch.zeros(edge_index.shape[1], device=device)
    get_nf        = _node_feature_getter(data)

    all_ret  = _asset_returns_from_prices(data["prices"], ckpt["universe"], device)
    prices_df = data["prices"]

    policy    = AssetWisePolicy(state_dim=config.embedding_dim, hidden=64).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr_drl)

    results: List[EpisodeResult] = []

    for ep_idx, ep_dates in enumerate(episode_dates_list[:n_drl_epochs]):
        # Warm-up encoder through pre-episode history
        encoder.reset_state()
        _warm_up_encoder(encoder, data, warm_dates, edge_index, edge_type_ids, edge_ts, get_nf, device)

        log_returns, log_probs = [], []
        entropy_sum  = 0.0
        weights_ok   = True

        policy.train()
        for d_idx in range(len(ep_dates) - 1):
            today    = ep_dates[d_idx]
            tomorrow = ep_dates[d_idx + 1]
            events   = data["events_by_date"].get(today, [])
            nf       = get_nf(today).to(device)
            t        = float(today) + 0.99

            with torch.no_grad():
                encoder.advance_day(events, nf, edge_index, edge_type_ids, edge_ts, t)
                Z = encoder.get_node_embeddings(nf, edge_index, edge_type_ids, edge_ts, t)
                # Z: (N, embedding_dim) -- per-asset temporal graph embedding

            weights = policy(Z.detach())  # (N,) -- policy is the learner
            weights_ok &= bool((weights >= -1e-6).all() and abs(weights.sum() - 1.0) < 1e-4)

            # Actual next-day returns
            tom_date = datetime.date.fromisoformat(_int_day_to_iso(tomorrow))
            if tom_date in prices_df.index:
                nr = prices_df.reindex(columns=ckpt["universe"]).loc[tom_date].ffill().fillna(0.0)
                next_ret = torch.tensor(nr.values, dtype=torch.float32, device=device)
            else:
                next_ret = torch.zeros(num_nodes, device=device)

            r = _portfolio_log_return(weights, next_ret)
            log_returns.append(r)
            log_probs.append(torch.log(weights + 1e-8).mean())
            entropy_sum += _entropy(weights)

        # REINFORCE update
        rewards = torch.stack([r.detach() for r in log_returns])
        baseline = rewards.mean()
        loss = -(torch.stack(log_probs) @ (rewards - baseline))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()

        cum = float(sum(r.item() for r in log_returns))
        results.append(EpisodeResult(
            episode_idx=ep_idx,
            condition="DyFO-DRL",
            cumulative_log_return=cum,
            sharpe=_sharpe([r.item() for r in log_returns]),
            mean_entropy=entropy_sum / max(1, len(log_returns)),
            weights_always_valid=weights_ok,
        ))
        print(f"  DyFO-DRL ep {ep_idx+1:3d}/{n_drl_epochs} | cum={cum:+.4f} sharpe={results[-1].sharpe:.2f}")

    return results, policy


# ─────────────────────────────────────────────────────────────────────────────
# Condition B: Raw-DRL (no graph)
# ─────────────────────────────────────────────────────────────────────────────

def _raw_state(prices_df, universe: List[str], today_date, window: int, device: torch.device) -> torch.Tensor:
    """Per-asset raw features: (return_t, 5d-mean, 5d-std) -> (N, 3)."""
    import pandas as pd
    prices = prices_df.reindex(columns=universe).ffill()
    loc = prices.index.get_loc(today_date)
    if loc < window:
        hist = prices.iloc[:loc + 1]
    else:
        hist = prices.iloc[loc - window: loc + 1]
    rets = hist.pct_change().dropna()
    if len(rets) == 0:
        return torch.zeros(len(universe), 3, device=device)
    latest   = torch.tensor(rets.iloc[-1].fillna(0.0).values,  dtype=torch.float32)
    momentum = torch.tensor(rets.mean().fillna(0.0).values,    dtype=torch.float32)
    vol      = torch.tensor(rets.std().fillna(0.0).clamp(lower=1e-6).values, dtype=torch.float32)
    return torch.stack([latest, momentum, vol], dim=-1).to(device)  # (N, 3)


def run_raw_drl_episodes(
    ckpt: dict,
    data: dict,
    episode_dates_list: List[List[int]],
    device: torch.device,
    n_drl_epochs: int,
    lr_drl: float,
    seed: int,
) -> tuple[List[EpisodeResult], AssetWisePolicy]:
    """DRL with raw price features — ablation of graph structure."""
    set_seed(seed + 1)
    universe  = ckpt["universe"]

    policy    = AssetWisePolicy(state_dim=3, hidden=64).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr_drl)

    results: List[EpisodeResult] = []
    prices_df = data["prices"].reindex(columns=universe).ffill()

    for ep_idx, ep_dates in enumerate(episode_dates_list[:n_drl_epochs]):
        log_returns, log_probs = [], []
        entropy_sum  = 0.0
        weights_ok   = True

        policy.train()
        for d_idx in range(len(ep_dates) - 1):
            today    = ep_dates[d_idx]
            tomorrow = ep_dates[d_idx + 1]

            today_date = datetime.date.fromisoformat(_int_day_to_iso(today))
            if today_date not in prices_df.index:
                continue

            Z = _raw_state(prices_df, universe, today_date, window=10, device=device)  # (N, 3)
            weights = policy(Z)
            weights_ok &= bool((weights >= -1e-6).all() and abs(weights.sum() - 1.0) < 1e-4)

            tom_date = datetime.date.fromisoformat(_int_day_to_iso(tomorrow))
            if tom_date in prices_df.index:
                nr = prices_df.loc[tom_date].pct_change().fillna(0.0)
                next_ret = torch.tensor(
                    prices_df.loc[tom_date].values / prices_df.loc[today_date].values - 1.0,
                    dtype=torch.float32, device=device,
                ).nan_to_num(0.0)
            else:
                next_ret = torch.zeros(len(universe), device=device)

            r = _portfolio_log_return(weights, next_ret)
            log_returns.append(r)
            log_probs.append(torch.log(weights + 1e-8).mean())
            entropy_sum += _entropy(weights)

        if not log_returns:
            continue

        rewards  = torch.stack([r.detach() for r in log_returns])
        baseline = rewards.mean()
        loss = -(torch.stack(log_probs) @ (rewards - baseline))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()

        cum = float(sum(r.item() for r in log_returns))
        results.append(EpisodeResult(
            episode_idx=ep_idx,
            condition="Raw-DRL",
            cumulative_log_return=cum,
            sharpe=_sharpe([r.item() for r in log_returns]),
            mean_entropy=entropy_sum / max(1, len(log_returns)),
            weights_always_valid=weights_ok,
        ))
        print(f"  Raw-DRL  ep {ep_idx+1:3d}/{n_drl_epochs} | cum={cum:+.4f} sharpe={results[-1].sharpe:.2f}")

    return results, policy


# ─────────────────────────────────────────────────────────────────────────────
# Condition C: EWMA-GMVP (no learning)
# ─────────────────────────────────────────────────────────────────────────────

def run_ewma_gmvp_episodes(
    ckpt: dict,
    data: dict,
    episode_dates_list: List[List[int]],
    device: torch.device,
    alpha: float = ALPHA_EWMA,
) -> List[EpisodeResult]:
    universe  = ckpt["universe"]
    prices_df = data["prices"].reindex(columns=universe).ffill()

    results: List[EpisodeResult] = []

    for ep_idx, ep_dates in enumerate(episode_dates_list):
        log_returns  = []
        entropy_sum  = 0.0
        weights_ok   = True

        for d_idx in range(len(ep_dates) - 1):
            today    = ep_dates[d_idx]
            tomorrow = ep_dates[d_idx + 1]

            today_date = datetime.date.fromisoformat(_int_day_to_iso(today))
            tom_date   = datetime.date.fromisoformat(_int_day_to_iso(tomorrow))

            if today_date not in prices_df.index or tom_date not in prices_df.index:
                continue

            # EWMA covariance from all history up to today
            loc  = prices_df.index.get_loc(today_date)
            hist = prices_df.iloc[:loc + 1].pct_change().dropna()
            if len(hist) < 5:
                continue
            cov = _ewma_cov(
                torch.tensor(hist.values, dtype=torch.float32),
                alpha=alpha,
            )
            weights  = _gmvp(cov)
            weights_ok &= bool((weights >= -1e-6).all() and abs(weights.sum() - 1.0) < 1e-4)

            next_ret = torch.tensor(
                prices_df.loc[tom_date].values / prices_df.loc[today_date].values - 1.0,
                dtype=torch.float32,
            ).nan_to_num(0.0)

            with torch.no_grad():
                r = _portfolio_log_return(weights, next_ret)
            log_returns.append(float(r))
            entropy_sum += _entropy(weights)

        cum = sum(log_returns)
        results.append(EpisodeResult(
            episode_idx=ep_idx,
            condition="EWMA-GMVP",
            cumulative_log_return=cum,
            sharpe=_sharpe(log_returns),
            mean_entropy=entropy_sum / max(1, len(log_returns)),
            weights_always_valid=weights_ok,
        ))
        print(f"  EWMA-GMVP ep {ep_idx+1:3d}/{len(episode_dates_list)} | cum={cum:+.4f} sharpe={results[-1].sharpe:.2f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Episode window builder
# ─────────────────────────────────────────────────────────────────────────────

def build_episodes(
    test_dates: List[int],
    episode_len: int,
    step: Optional[int] = None,
) -> List[List[int]]:
    """Slice test_dates into rolling windows of length episode_len."""
    if step is None:
        step = episode_len // 2
    episodes = []
    i = 0
    while i + episode_len <= len(test_dates):
        episodes.append(test_dates[i: i + episode_len])
        i += step
    return episodes


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def _report(results_by_condition: dict) -> dict:
    summary = {}
    for cond, results in results_by_condition.items():
        n = len(results)
        if n == 0:
            continue
        cum  = [r.cumulative_log_return for r in results]
        sh   = [r.sharpe for r in results if math.isfinite(r.sharpe)]
        entr = [r.mean_entropy for r in results]
        all_valid = all(r.weights_always_valid for r in results)
        summary[cond] = {
            "n_episodes": n,
            "mean_cum_log_ret": float(sum(cum) / n),
            "mean_sharpe":      float(sum(sh) / len(sh)) if sh else float("nan"),
            "mean_entropy":     float(sum(entr) / n),
            "all_weights_valid": all_valid,
            "episodes": [r._asdict() for r in results],
        }
    return summary


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 72)
    print(f"{'Condition':<20} {'N':>4} {'Mean CumRet':>12} {'Mean Sharpe':>12} {'Entropy':>9}")
    print("=" * 72)
    for cond, s in summary.items():
        print(
            f"  {cond:<18} {s['n_episodes']:>4} "
            f"{s['mean_cum_log_ret']:>+12.4f} "
            f"{s['mean_sharpe']:>12.3f} "
            f"{s['mean_entropy']:>9.3f}"
        )
    print("=" * 72)
    print(
        "\nNote: with random-init DRL policy this is an interface/plumbing test.\n"
        "Load a converged DyFO checkpoint and run more episodes for real comparison."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DyFO and run DRL portfolio episodes (3 conditions)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--phase", choices=["train", "drl", "all"], default="all",
        help="'train': only Phase 1 (save checkpoint). "
             "'drl': only Phase 2 (load checkpoint + run episodes). "
             "'all': both phases end-to-end.",
    )
    parser.add_argument("--epochs",       type=int,   default=15,  help="TGAT training epochs.")
    parser.add_argument("--drl_episodes", type=int,   default=20,  help="Number of DRL episodes per condition.")
    parser.add_argument("--episode_len",  type=int,   default=EPISODE_LEN, help="Trading days per episode.")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--lr",           type=float, default=TGN_LR)
    parser.add_argument("--lr_drl",       type=float, default=3e-4)
    parser.add_argument(
        "--checkpoint",
        default=str(RESULTS_DIR / "dyfo_portfolio_ckpt.pt"),
        help="Path to save/load DyFO checkpoint.",
    )
    parser.add_argument(
        "--drl_policy_out",
        default=str(RESULTS_DIR / "dyfo_drl_policy.pt"),
        help="Path to save trained DRL policy.",
    )
    parser.add_argument(
        "--report_out",
        default=str(RESULTS_DIR / "dyfo_portfolio_report.json"),
        help="Path to save episode report JSON.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging("dyfo.portfolio", log_to_file=False)
    ckpt_path = Path(args.checkpoint)

    config      = DyFOConfig(model_variant="tgat")
    data_config = DataConfig(
        tickers=UNIVERSE,
        benchmark_ticker="SPY",
        start_date=DATA_START,
        end_date=DATA_END,
    )

    # ── prepare data (cached) ────────────────────────────────────────────────
    print(f"Universe ({len(UNIVERSE)} assets): {UNIVERSE}")
    print("Loading / preparing market data (this may take a few minutes) ...")
    data = load_or_prepare_data(
        tickers=UNIVERSE,
        start=DATA_START,
        end=DATA_END,
        benchmark="SPY",
        config=config,
        data_config=data_config,
        logger=logger,
    )

    all_dates   = data["sorted_dates"]
    train_dates = _slice_dates(all_dates, DATA_START,  TRAIN_END)
    val_dates   = _slice_dates(all_dates, TRAIN_END,   VAL_END)
    test_dates  = _slice_dates(all_dates, TEST_START,  TEST_END)

    print(
        f"Split: train={len(train_dates)}d  val={len(val_dates)}d  "
        f"test={len(test_dates)}d"
    )

    # ── Phase 1: train ───────────────────────────────────────────────────────
    if args.phase in ("train", "all"):
        print(f"\n[Phase 1] Training TGAT (epochs={args.epochs}) ...")
        result = train_dyfo(
            data=data,
            num_nodes=len(UNIVERSE),
            train_dates=train_dates,
            val_dates=val_dates,
            num_epochs=args.epochs,
            lr=args.lr,
            patience=TGN_PATIENCE,
            seed=args.seed,
            device=device,
        )
        save_checkpoint(
            path=ckpt_path,
            train_result=result,
            universe=UNIVERSE,
            ticker_to_idx=data["ticker_to_idx"],
            config=config,
        )
        print(
            f"  Best val R2={result['best_val_r2']:.4f} "
            f"at epoch {result['best_epoch']}"
        )

    # ── Phase 2: DRL episodes ────────────────────────────────────────────────
    if args.phase in ("drl", "all"):
        print(f"\n[Phase 2] DRL episodes ({args.drl_episodes} per condition) ...")
        ckpt = load_checkpoint(ckpt_path)

        episodes = build_episodes(test_dates, episode_len=args.episode_len)
        if not episodes:
            raise RuntimeError(
                f"No full episodes in test window "
                f"({len(test_dates)} days, episode_len={args.episode_len}). "
                "Shorten --episode_len or extend the test window."
            )

        # Cycle episodes if we need more than available
        ep_list = []
        while len(ep_list) < args.drl_episodes:
            ep_list.extend(episodes)
        ep_list = ep_list[:args.drl_episodes]

        print(f"  {len(episodes)} distinct episode windows -> "
              f"cycling to {len(ep_list)} for DRL training")

        # Warm-up dates = last 20 days of val (fills encoder buffers)
        warm_dates = val_dates[-20:] if len(val_dates) >= 20 else val_dates

        print("\n  [A] DyFO-DRL ...")
        dyfo_results, dyfo_policy = run_dyfo_drl_episodes(
            ckpt=ckpt,
            data=data,
            episode_dates_list=ep_list,
            warm_dates=warm_dates,
            config=config,
            device=device,
            n_drl_epochs=args.drl_episodes,
            lr_drl=args.lr_drl,
            seed=args.seed,
        )

        print("\n  [B] Raw-DRL ...")
        raw_results, raw_policy = run_raw_drl_episodes(
            ckpt=ckpt,
            data=data,
            episode_dates_list=ep_list,
            device=device,
            n_drl_epochs=args.drl_episodes,
            lr_drl=args.lr_drl,
            seed=args.seed,
        )

        print("\n  [C] EWMA-GMVP ...")
        ewma_results = run_ewma_gmvp_episodes(
            ckpt=ckpt,
            data=data,
            episode_dates_list=ep_list,
            device=device,
        )

        # ── Save DRL policy ────────────────────────────────────────────────
        policy_path = Path(args.drl_policy_out)
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "dyfo_policy": dyfo_policy.state_dict(),
            "raw_policy":  raw_policy.state_dict(),
            "embedding_dim": ckpt["embedding_dim"],
            "num_nodes":     ckpt["num_nodes"],
            "universe":      ckpt["universe"],
        }, policy_path)
        print(f"\nDRL policies saved -> {policy_path}")

        # ── Report ─────────────────────────────────────────────────────────
        summary = _report({
            "DyFO-DRL":  dyfo_results,
            "Raw-DRL":   raw_results,
            "EWMA-GMVP": ewma_results,
        })
        print_summary(summary)

        report_path = Path(args.report_out)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Report saved -> {report_path}")


if __name__ == "__main__":
    main()
