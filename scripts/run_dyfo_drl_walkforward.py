#!/usr/bin/env python3
"""Risk-aware DyFO-DRL walk-forward evaluation for portfolio research.

This script is the publication-oriented successor to the original
``run_dyfo_drl_walkforward.py`` runner. It keeps the same causal walk-forward
idea and strategy set, but adds the machinery needed for a defensible
portfolio-RL experiment:

* Risk-regularized DyFO and Raw DRL policies through a composable
  ``RiskRegularizer``.
* Penalization for turnover, volatility, CVaR, max drawdown and CDaR.
* Reward objectives based on log-return, Sharpe, Sortino or Calmar.
* Hyperparameter grid search over turnover penalty, risk aversion and drawdown
  penalty, with one artifact directory per combination.
* Matched out-of-sample evaluation across DyFO-DRL, DyFO-DRL+, Raw-DRL,
  Raw-DRL+, EWMA-GMVP and EqualWeight.
* Regime/subperiod analysis, asset contribution, allocation attribution and
  automatic plots for allocation, regime contribution heatmaps and cumulative
  performance.
* YAML or CLI configuration, robust logging, optional TensorBoard logging, and
  reproducible outputs.

The official no-leakage mode is ``--checkpoint_mode causal``. For quick smoke
tests and notebook iteration, ``--checkpoint_mode reuse`` reuses an existing
TGAT checkpoint and marks the resulting report as non-causal for TGAT.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import logging
import math
import random
import re
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:  # YAML is optional so the script still runs in lean environments.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - exercised only when PyYAML is absent.
    yaml = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - tensorboard is optional.
    SummaryWriter = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting remains optional.
    plt = None

try:
    import seaborn as sns
except Exception:  # pragma: no cover - heatmaps fall back to matplotlib.
    sns = None

from dyfo.config import DataConfig, DyFOConfig
from dyfo.core.model_variants import build_encoder
from dyfo.logging_utils import RESULTS_DIR, setup_logging
from scripts.run_bootstrap_eval_v5 import TGN_LR, TGN_PATIENCE, load_or_prepare_data
from scripts.train_dyfo_portfolio import (
    ALPHA_EWMA,
    DATA_END,
    DATA_START,
    EPISODE_LEN,
    UNIVERSE,
    AssetWisePolicy,
    AttentionPortfolioPolicy,
    _entropy,
    _ewma_cov,
    _gmvp,
    _int_day_to_iso,
    _node_feature_getter,
    _portfolio_log_return,
    _raw_state,
    _sharpe,
    build_episodes,
    load_checkpoint,
    save_checkpoint,
    train_dyfo,
)
from scripts.train_link_prediction import set_seed


CONDITIONS = [
    "DyFO-DRL",
    "DyFO-DRL+",
    "Raw-DRL",
    "Raw-DRL+",
    "EWMA-GMVP",
    "EqualWeight",
]

RISK_AWARE_CONDITIONS = {"DyFO-DRL", "DyFO-DRL+", "Raw-DRL", "Raw-DRL+"}


# ---------------------------------------------------------------------------
# 1. Configuration and hyperparameters
# ---------------------------------------------------------------------------


@dataclass
class RiskRegularizerConfig:
    """Composable risk/reward regularization parameters for RL policies."""

    reward_objective: str = "sharpe"  # log_return, sharpe, sortino, calmar
    turnover_penalty: float = 0.0
    risk_aversion: float = 0.0
    volatility_target: Optional[float] = None
    volatility_penalty_weight: float = 0.0
    cvar_alpha: float = 0.05
    cvar_penalty_weight: float = 0.0
    drawdown_penalty_weight: float = 0.0
    cdar_alpha: float = 0.20
    cdar_penalty_weight: float = 0.0
    max_weight: Optional[float] = None
    max_weight_penalty_weight: float = 0.0
    entropy_bonus_weight: float = 0.0


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration.

    Defaults intentionally mirror the previous official command:

    ``--checkpoint_mode causal --train_days 1000 --val_days 250
    --test_days 125 --step_days 125 --episode_len 60 --drl_episodes 30
    --seeds 42,123,456 --epochs 8 --n_bootstrap 1000``.
    """

    start: str = DATA_START
    end: str = DATA_END
    train_days: int = 1000
    val_days: int = 250
    test_days: int = 125
    step_days: int = 125
    max_windows: Optional[int] = None
    episode_len: int = EPISODE_LEN
    train_episode_step: Optional[int] = None
    test_episode_step: Optional[int] = None
    drl_episodes: int = 30
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    epochs: int = 8
    lr: float = TGN_LR
    lr_drl: float = 3e-4
    n_heads: int = 4
    finetune_encoder: bool = False
    warmup_days: int = 20
    alpha_ewma: float = ALPHA_EWMA
    n_bootstrap: int = 1000
    bootstrap_block_len: int = 20
    bootstrap_seed: int = 2026
    checkpoint_mode: str = "causal"  # causal, reuse
    checkpoint: str = str(RESULTS_DIR / "dyfo_portfolio_ckpt.pt")
    force_retrain: bool = False
    cpu: bool = False
    out_dir: str = str(RESULTS_DIR / "dyfo_drl_walkforward")
    risk: RiskRegularizerConfig = field(default_factory=RiskRegularizerConfig)
    turnover_grid: List[float] = field(default_factory=lambda: [0.0])
    risk_aversion_grid: List[float] = field(default_factory=lambda: [0.0])
    drawdown_penalty_grid: List[float] = field(default_factory=lambda: [0.0])
    max_grid_runs: Optional[int] = None
    enable_tensorboard: bool = False
    save_models: bool = False
    make_plots: bool = True
    log_level: str = "INFO"
    raw_optimizer: str = "direct"  # direct, ppo
    raw_ppo_clip: float = 0.2
    raw_ppo_epochs: int = 4
    raw_ppo_value_coef: float = 0.5
    raw_ppo_concentration: float = 30.0


@dataclass
class WindowSpec:
    window_idx: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str
    train_dates: List[int]
    val_dates: List[int]
    test_dates: List[int]


@dataclass
class StepRecord:
    grid_id: str
    window_idx: int
    seed: int
    episode_idx: int
    condition: str
    date: str
    asset: str
    weight: float
    asset_return: float
    weighted_return: float
    turnover_contribution: float


@dataclass
class EvalRecord:
    grid_id: str
    window_idx: int
    seed: int
    episode_idx: int
    condition: str
    test_start: str
    test_end: str
    cumulative_log_return: float
    sharpe: float
    sortino: float
    calmar: float
    cvar_5: float
    realized_volatility: float
    max_drawdown: float
    cdar: float
    mean_entropy: float
    mean_turnover: float
    weights_always_valid: bool
    daily_log_returns: List[float]


# ---------------------------------------------------------------------------
# 2. Risk regularization
# ---------------------------------------------------------------------------


class RiskRegularizer:
    """Combines risk and drawdown penalties into an RL reward.

    The class operates on differentiable tensors during training. Metrics are
    also exposed as plain floats for reporting. Penalties are intentionally
    simple and stable because this script must run dozens of windows/seeds in
    a few hours, not perform slow policy-gradient experimentation.
    """

    def __init__(self, config: RiskRegularizerConfig):
        self.config = config

    def transform_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Softly constrain actions before portfolio returns are computed."""
        if self.config.max_weight is None:
            return weights
        capped = torch.clamp(weights, max=float(self.config.max_weight))
        total = capped.sum().clamp(min=1e-8)
        return capped / total

    def objective(self, log_returns: torch.Tensor) -> torch.Tensor:
        if log_returns.numel() == 0:
            return torch.tensor(0.0, device=log_returns.device)
        obj = self.config.reward_objective.lower()
        mean = log_returns.mean()
        std = log_returns.std(unbiased=False).clamp(min=1e-8)
        if obj == "log_return":
            return log_returns.sum()
        if obj == "sharpe":
            return mean / std * math.sqrt(252)
        if obj == "sortino":
            downside = torch.clamp(-log_returns, min=0.0)
            downside_dev = torch.sqrt((downside ** 2).mean()).clamp(min=1e-8)
            return mean / downside_dev * math.sqrt(252)
        if obj == "calmar":
            mdd = self.max_drawdown(log_returns).clamp(min=1e-8)
            annual_ret = mean * 252
            return annual_ret / mdd
        raise ValueError(f"Unsupported reward objective: {self.config.reward_objective}")

    def penalty(
        self,
        log_returns: torch.Tensor,
        weights: Sequence[torch.Tensor],
        turnovers: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if log_returns.numel() == 0:
            return torch.tensor(0.0, device=log_returns.device)

        cfg = self.config
        penalty = torch.tensor(0.0, device=log_returns.device)

        if cfg.turnover_penalty and turnovers:
            penalty = penalty + float(cfg.turnover_penalty) * torch.stack(list(turnovers)).mean()

        if cfg.risk_aversion:
            penalty = penalty + float(cfg.risk_aversion) * log_returns.var(unbiased=False)

        if cfg.volatility_target is not None and cfg.volatility_penalty_weight:
            realized = log_returns.std(unbiased=False) * math.sqrt(252)
            excess = torch.clamp(realized - float(cfg.volatility_target), min=0.0)
            penalty = penalty + float(cfg.volatility_penalty_weight) * excess.pow(2)

        if cfg.cvar_penalty_weight:
            penalty = penalty + float(cfg.cvar_penalty_weight) * self.cvar_loss(log_returns)

        if cfg.drawdown_penalty_weight:
            penalty = penalty + float(cfg.drawdown_penalty_weight) * self.max_drawdown(log_returns)

        if cfg.cdar_penalty_weight:
            penalty = penalty + float(cfg.cdar_penalty_weight) * self.cdar(log_returns)

        if cfg.max_weight is not None and cfg.max_weight_penalty_weight and weights:
            over = [torch.clamp(w.max() - float(cfg.max_weight), min=0.0).pow(2) for w in weights]
            penalty = penalty + float(cfg.max_weight_penalty_weight) * torch.stack(over).mean()

        if cfg.entropy_bonus_weight and weights:
            entropy = torch.stack([-(w * (w + 1e-8).log()).sum() for w in weights]).mean()
            penalty = penalty - float(cfg.entropy_bonus_weight) * entropy

        return penalty

    def reward(
        self,
        log_returns: torch.Tensor,
        weights: Sequence[torch.Tensor],
        turnovers: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        return self.objective(log_returns) - self.penalty(log_returns, weights, turnovers)

    def metric_dict(self, daily_log_returns: List[float]) -> dict:
        if not daily_log_returns:
            return {
                "sortino": float("nan"),
                "calmar": float("nan"),
                "cvar_5": float("nan"),
                "realized_volatility": float("nan"),
                "max_drawdown": float("nan"),
                "cdar": float("nan"),
            }
        returns = torch.tensor(daily_log_returns, dtype=torch.float32)
        mdd = float(self.max_drawdown(returns))
        cdar = float(self.cdar(returns))
        mean = float(returns.mean())
        std = float(returns.std(unbiased=False))
        downside = torch.clamp(-returns, min=0.0)
        downside_dev = float(torch.sqrt((downside ** 2).mean()).clamp(min=1e-8))
        sortino = mean / downside_dev * math.sqrt(252)
        calmar = (mean * 252) / max(abs(mdd), 1e-8)
        return {
            "sortino": float(sortino),
            "calmar": float(calmar),
            "cvar_5": float(self.cvar_loss(returns, alpha=0.05)),
            "realized_volatility": float(std * math.sqrt(252)),
            "max_drawdown": mdd,
            "cdar": cdar,
        }

    @staticmethod
    def max_drawdown(log_returns: torch.Tensor) -> torch.Tensor:
        equity = torch.exp(torch.cumsum(log_returns, dim=0))
        running_peak = torch.cummax(equity, dim=0).values.clamp(min=1e-8)
        dd = 1.0 - equity / running_peak
        return dd.max()

    def cdar(self, log_returns: torch.Tensor, alpha: Optional[float] = None) -> torch.Tensor:
        equity = torch.exp(torch.cumsum(log_returns, dim=0))
        running_peak = torch.cummax(equity, dim=0).values.clamp(min=1e-8)
        dd = 1.0 - equity / running_peak
        if dd.numel() == 0:
            return torch.tensor(0.0, device=log_returns.device)
        frac = float(self.config.cdar_alpha if alpha is None else alpha)
        k = max(1, int(math.ceil(frac * dd.numel())))
        return torch.topk(dd, k=k).values.mean()

    def cvar_loss(self, log_returns: torch.Tensor, alpha: Optional[float] = None) -> torch.Tensor:
        losses = -log_returns
        if losses.numel() == 0:
            return torch.tensor(0.0, device=log_returns.device)
        frac = float(self.config.cvar_alpha if alpha is None else alpha)
        k = max(1, int(math.ceil(frac * losses.numel())))
        return torch.topk(losses, k=k).values.mean()


# ---------------------------------------------------------------------------
# 3. Environment helpers
# ---------------------------------------------------------------------------


class PortfolioEpisodeEnv:
    """Thin portfolio environment for one episode of market dates.

    The environment deliberately stays minimal: the caller supplies a policy and
    state builder, while this class handles returns, weight validation, turnover
    and attribution rows in one place.
    """

    def __init__(self, prices_df: pd.DataFrame, universe: List[str], device: torch.device):
        self.prices_df = prices_df.reindex(columns=universe).ffill()
        self.rets_df = self.prices_df.pct_change().fillna(0.0)
        self.universe = universe
        self.device = device

    def next_return(self, today: int, tomorrow: int) -> torch.Tensor:
        today_ts = pd.Timestamp(_int_day_to_iso(today))
        tom_ts = pd.Timestamp(_int_day_to_iso(tomorrow))
        if tom_ts in self.rets_df.index:
            return torch.tensor(self.rets_df.loc[tom_ts].values, dtype=torch.float32, device=self.device).nan_to_num(0.0)
        if today_ts in self.prices_df.index and tom_ts in self.prices_df.index:
            return torch.tensor(
                self.prices_df.loc[tom_ts].values / self.prices_df.loc[today_ts].values - 1.0,
                dtype=torch.float32,
                device=self.device,
            ).nan_to_num(0.0)
        return torch.zeros(len(self.universe), dtype=torch.float32, device=self.device)


def _parse_seeds(raw: str | Sequence[int]) -> List[int]:
    if isinstance(raw, str):
        return [int(x.strip()) for x in raw.replace(";", ",").split(",") if x.strip()]
    return [int(x) for x in raw]


def _parse_float_grid(raw: str | Sequence[float]) -> List[float]:
    if isinstance(raw, str):
        return [float(x.strip()) for x in raw.replace(";", ",").split(",") if x.strip()]
    return [float(x) for x in raw]


def _slug_float(x: float) -> str:
    text = f"{x:g}".replace("-", "m").replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_]+", "_", text)


def _grid_id(turnover: float, risk: float, drawdown: float) -> str:
    return f"turnover_{_slug_float(turnover)}_risk_{_slug_float(risk)}_drawdown_{_slug_float(drawdown)}"


def _date(day: int) -> dt.date:
    return dt.date.fromisoformat(_int_day_to_iso(day))


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _build_walk_forward_windows(
    all_dates: List[int],
    train_days: int,
    val_days: int,
    test_days: int,
    step_days: int,
    max_windows: Optional[int],
) -> List[WindowSpec]:
    total = train_days + val_days + test_days
    windows: List[WindowSpec] = []
    start = 0
    while start + total <= len(all_dates):
        train = all_dates[start: start + train_days]
        val = all_dates[start + train_days: start + train_days + val_days]
        test = all_dates[start + train_days + val_days: start + total]
        windows.append(WindowSpec(
            window_idx=len(windows),
            train_start=_int_day_to_iso(train[0]),
            train_end=_int_day_to_iso(train[-1]),
            val_start=_int_day_to_iso(val[0]),
            val_end=_int_day_to_iso(val[-1]),
            test_start=_int_day_to_iso(test[0]),
            test_end=_int_day_to_iso(test[-1]),
            train_dates=train,
            val_dates=val,
            test_dates=test,
        ))
        if max_windows is not None and len(windows) >= max_windows:
            break
        start += step_days
    return windows


def _cycle_episodes(episodes: List[List[int]], n: int) -> List[List[int]]:
    if not episodes:
        return []
    out: List[List[int]] = []
    while len(out) < n:
        out.extend(episodes)
    return out[:n]


def _episode_regime(start_iso: str, end_iso: str, sp500_dd: Optional[float] = None) -> str:
    start = pd.Timestamp(start_iso)
    end = pd.Timestamp(end_iso)
    if end <= pd.Timestamp("2020-02-29"):
        return "pre_covid"
    if start <= pd.Timestamp("2020-12-31") and end >= pd.Timestamp("2020-03-01"):
        return "covid"
    if start <= pd.Timestamp("2023-12-31") and end >= pd.Timestamp("2021-01-01"):
        return "rates_inflation"
    if start >= pd.Timestamp("2024-01-01"):
        return "post_rates"
    if sp500_dd is not None and sp500_dd <= -0.20:
        return "bear_market"
    return "bull_market"


# ---------------------------------------------------------------------------
# 4. DyFO/Raw agents and policy training
# ---------------------------------------------------------------------------


def _prepare_checkpoint(
    *,
    cfg: ExperimentConfig,
    data: dict,
    config: DyFOConfig,
    window: WindowSpec,
    seed: int,
    device: torch.device,
    out_dir: Path,
    logger: logging.Logger,
) -> dict:
    if cfg.checkpoint_mode == "reuse":
        return load_checkpoint(Path(cfg.checkpoint))

    ckpt_path = out_dir / "checkpoints" / f"window_{window.window_idx:02d}_seed_{seed}.pt"
    if ckpt_path.exists() and not cfg.force_retrain:
        return load_checkpoint(ckpt_path)

    logger.info(
        "Training causal TGAT | window=%s seed=%s train=%s..%s val=%s..%s",
        window.window_idx,
        seed,
        window.train_start,
        window.train_end,
        window.val_start,
        window.val_end,
    )
    result = train_dyfo(
        data=data,
        num_nodes=len(UNIVERSE),
        train_dates=window.train_dates,
        val_dates=window.val_dates,
        num_epochs=cfg.epochs,
        lr=cfg.lr,
        patience=TGN_PATIENCE,
        seed=seed,
        device=device,
    )
    save_checkpoint(
        path=ckpt_path,
        train_result=result,
        universe=UNIVERSE,
        ticker_to_idx=data["ticker_to_idx"],
        config=config,
    )
    return load_checkpoint(ckpt_path)


def _build_dyfo_components(ckpt: dict, config: DyFOConfig, cfg: ExperimentConfig, device: torch.device, improved: bool):
    encoder = build_encoder(config, ckpt["num_nodes"], variant="tgat").to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    if improved:
        policy = AttentionPortfolioPolicy(config.embedding_dim, n_heads=cfg.n_heads, hidden=64).to(device)
    else:
        policy = AssetWisePolicy(config.embedding_dim, hidden=64).to(device)
    return encoder, policy


def _build_raw_policy(cfg: ExperimentConfig, device: torch.device, improved: bool):
    if improved:
        return AttentionPortfolioPolicy(3, n_heads=cfg.n_heads, hidden=64).to(device)
    return AssetWisePolicy(3, hidden=64).to(device)


class RawValueHead(nn.Module):
    """Value baseline for raw-feature PPO over flattened per-asset state."""

    def __init__(self, n_assets: int, n_features: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_assets * n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.reshape(1, -1)).squeeze()


def _raw_policy_log_prob(
    policy: nn.Module,
    z: torch.Tensor,
    regularizer: RiskRegularizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = regularizer.transform_weights(policy(z))
    log_prob = torch.log(weights + 1e-8).mean()
    return weights, log_prob


def _raw_policy_distribution(
    policy: nn.Module,
    z: torch.Tensor,
    regularizer: RiskRegularizer,
    total_concentration: float,
) -> tuple[torch.distributions.Dirichlet, torch.Tensor]:
    mean_weights = regularizer.transform_weights(policy(z))
    concentration = (mean_weights * float(total_concentration)).clamp(min=1e-3)
    return torch.distributions.Dirichlet(concentration), mean_weights


def _warm_encoder(encoder, data, warm_dates, edge_index, edge_type_ids, edge_ts, get_nf, device) -> None:
    encoder.eval()
    with torch.no_grad():
        for d in warm_dates:
            nf = get_nf(d).to(device)
            events = data["events_by_date"].get(d, [])
            encoder.advance_day(events, nf, edge_index, edge_type_ids, edge_ts, float(d) + 0.99)


def train_dyfo_policy(
    *,
    ckpt: dict,
    data: dict,
    config: DyFOConfig,
    cfg: ExperimentConfig,
    episodes: List[List[int]],
    warm_dates: List[int],
    device: torch.device,
    seed: int,
    regularizer: RiskRegularizer,
    improved: bool,
    label: str,
    writer=None,
):
    set_seed(seed)
    encoder, policy = _build_dyfo_components(ckpt, config, cfg, device, improved=improved)
    graph = data["graph"]
    edge_index = graph.get_full_edge_index().to(device)
    edge_type_ids = graph.get_edge_type_ids().to(device)
    edge_ts = torch.zeros(edge_index.shape[1], device=device)
    get_nf = _node_feature_getter(data)
    env = PortfolioEpisodeEnv(data["prices"], ckpt["universe"], device)

    params = [{"params": policy.parameters(), "lr": cfg.lr_drl}]
    if cfg.finetune_encoder:
        params.append({"params": encoder.parameters(), "lr": cfg.lr_drl * 0.01})
    optimizer = optim.Adam(params)

    for ep_idx, ep_dates in enumerate(episodes):
        encoder.reset_state()
        if cfg.finetune_encoder:
            encoder.train()
        else:
            encoder.eval()
        _warm_encoder(encoder, data, warm_dates, edge_index, edge_type_ids, edge_ts, get_nf, device)
        policy.train()

        log_returns: List[torch.Tensor] = []
        weights_seq: List[torch.Tensor] = []
        turnovers: List[torch.Tensor] = []
        prev_w = None

        for today, tomorrow in zip(ep_dates[:-1], ep_dates[1:]):
            nf = get_nf(today).to(device)
            events = data["events_by_date"].get(today, [])
            t = float(today) + 0.99
            if cfg.finetune_encoder:
                encoder.advance_day(events, nf, edge_index, edge_type_ids, edge_ts, t)
                z = encoder.get_node_embeddings(nf, edge_index, edge_type_ids, edge_ts, t)
            else:
                with torch.no_grad():
                    encoder.advance_day(events, nf, edge_index, edge_type_ids, edge_ts, t)
                    z = encoder.get_node_embeddings(nf, edge_index, edge_type_ids, edge_ts, t)
                z = z.detach()

            weights = regularizer.transform_weights(policy(z))
            next_ret = env.next_return(today, tomorrow)
            log_returns.append(_portfolio_log_return(weights, next_ret))
            weights_seq.append(weights)
            if prev_w is not None:
                turnovers.append(torch.abs(weights - prev_w).sum())
            prev_w = weights.detach()

        if not log_returns:
            continue
        returns_t = torch.stack(log_returns)
        reward = regularizer.reward(returns_t, weights_seq, turnovers)
        loss = -reward
        optimizer.zero_grad()
        loss.backward()
        params_to_clip = list(policy.parameters()) + (list(encoder.parameters()) if cfg.finetune_encoder else [])
        nn.utils.clip_grad_norm_(params_to_clip, max_norm=0.5)
        optimizer.step()
        if cfg.finetune_encoder:
            encoder.detach_state()
        if writer is not None:
            writer.add_scalar(f"train/{label}/reward", float(reward.detach()), ep_idx)
            writer.add_scalar(f"train/{label}/loss", float(loss.detach()), ep_idx)

    return encoder, policy


def train_raw_policy(
    *,
    ckpt: dict,
    data: dict,
    cfg: ExperimentConfig,
    episodes: List[List[int]],
    device: torch.device,
    seed: int,
    regularizer: RiskRegularizer,
    improved: bool,
    label: str,
    writer=None,
):
    set_seed(seed + (17 if improved else 11))
    policy = _build_raw_policy(cfg, device, improved=improved)
    value_head = RawValueHead(n_assets=len(ckpt["universe"])).to(device) if cfg.raw_optimizer == "ppo" else None
    params = list(policy.parameters()) + (list(value_head.parameters()) if value_head is not None else [])
    optimizer = optim.Adam(params, lr=cfg.lr_drl)
    env = PortfolioEpisodeEnv(data["prices"], ckpt["universe"], device)
    prices_df = env.prices_df

    for ep_idx, ep_dates in enumerate(episodes):
        policy.train()
        if value_head is not None:
            value_head.train()
        log_returns: List[torch.Tensor] = []
        weights_seq: List[torch.Tensor] = []
        turnovers: List[torch.Tensor] = []
        states: List[torch.Tensor] = []
        actions: List[torch.Tensor] = []
        old_log_probs: List[torch.Tensor] = []
        prev_w = None
        for today, tomorrow in zip(ep_dates[:-1], ep_dates[1:]):
            today_ts = pd.Timestamp(_int_day_to_iso(today))
            if today_ts not in prices_df.index:
                continue
            z = _raw_state(prices_df, ckpt["universe"], today_ts, window=10, device=device)
            if cfg.raw_optimizer == "ppo":
                dist, _mean_weights = _raw_policy_distribution(
                    policy,
                    z,
                    regularizer,
                    cfg.raw_ppo_concentration,
                )
                action = dist.rsample().detach()
                weights = regularizer.transform_weights(action)
                log_prob = dist.log_prob(action)
                actions.append(action)
            else:
                weights, log_prob = _raw_policy_log_prob(policy, z, regularizer)
            next_ret = env.next_return(today, tomorrow)
            log_returns.append(_portfolio_log_return(weights, next_ret))
            weights_seq.append(weights)
            states.append(z.detach())
            old_log_probs.append(log_prob.detach())
            if prev_w is not None:
                turnovers.append(torch.abs(weights - prev_w).sum())
            prev_w = weights.detach()

        if not log_returns:
            continue
        returns_t = torch.stack(log_returns)
        reward = regularizer.reward(returns_t, weights_seq, turnovers)
        if cfg.raw_optimizer == "ppo":
            if value_head is None:
                raise RuntimeError("raw_optimizer='ppo' requires a value head")
            old_log_probs_t = torch.stack(old_log_probs)
            reward_target = reward.detach()
            last_loss = None
            for _ in range(max(1, int(cfg.raw_ppo_epochs))):
                new_log_probs = []
                values = []
                for z, action in zip(states, actions):
                    dist, _mean_weights = _raw_policy_distribution(
                        policy,
                        z,
                        regularizer,
                        cfg.raw_ppo_concentration,
                    )
                    log_prob = dist.log_prob(action)
                    new_log_probs.append(log_prob)
                    values.append(value_head(z))
                new_log_probs_t = torch.stack(new_log_probs)
                values_t = torch.stack(values).view(-1)
                targets = reward_target.to(values_t.device).expand_as(values_t)
                advantages = (targets - values_t.detach())
                if advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp(min=1e-8)
                ratio = torch.exp(new_log_probs_t - old_log_probs_t)
                clipped_ratio = torch.clamp(ratio, 1.0 - cfg.raw_ppo_clip, 1.0 + cfg.raw_ppo_clip)
                policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
                value_loss = nn.functional.mse_loss(values_t, targets)
                loss = policy_loss + float(cfg.raw_ppo_value_coef) * value_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, max_norm=0.5)
                optimizer.step()
                last_loss = loss
            loss = last_loss if last_loss is not None else -reward
        else:
            loss = -reward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()
        if writer is not None:
            writer.add_scalar(f"train/{label}/reward", float(reward.detach()), ep_idx)
            writer.add_scalar(f"train/{label}/loss", float(loss.detach()), ep_idx)

    return policy


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------


def _record_eval(
    *,
    grid_id: str,
    window_idx: int,
    seed: int,
    episode_idx: int,
    condition: str,
    ep_dates: List[int],
    daily: List[float],
    entropies: List[float],
    turnovers: List[float],
    weights_ok: bool,
    regularizer: RiskRegularizer,
) -> EvalRecord:
    metrics = regularizer.metric_dict(daily)
    return EvalRecord(
        grid_id=grid_id,
        window_idx=window_idx,
        seed=seed,
        episode_idx=episode_idx,
        condition=condition,
        test_start=_int_day_to_iso(ep_dates[0]),
        test_end=_int_day_to_iso(ep_dates[-1]),
        cumulative_log_return=float(sum(daily)),
        sharpe=_sharpe(daily),
        sortino=metrics["sortino"],
        calmar=metrics["calmar"],
        cvar_5=metrics["cvar_5"],
        realized_volatility=metrics["realized_volatility"],
        max_drawdown=metrics["max_drawdown"],
        cdar=metrics["cdar"],
        mean_entropy=_mean(entropies),
        mean_turnover=_mean(turnovers) if turnovers else 0.0,
        weights_always_valid=weights_ok,
        daily_log_returns=[float(x) for x in daily],
    )


def _append_step_rows(
    rows: List[StepRecord],
    *,
    grid_id: str,
    window_idx: int,
    seed: int,
    episode_idx: int,
    condition: str,
    date_iso: str,
    universe: List[str],
    weights: torch.Tensor,
    next_ret: torch.Tensor,
    prev_w: Optional[torch.Tensor],
) -> None:
    weights_cpu = weights.detach().cpu()
    ret_cpu = next_ret.detach().cpu()
    prev_cpu = prev_w.detach().cpu() if prev_w is not None else torch.zeros_like(weights_cpu)
    turnover = torch.abs(weights_cpu - prev_cpu)
    for asset, w, r, to in zip(universe, weights_cpu.tolist(), ret_cpu.tolist(), turnover.tolist()):
        rows.append(StepRecord(
            grid_id=grid_id,
            window_idx=window_idx,
            seed=seed,
            episode_idx=episode_idx,
            condition=condition,
            date=date_iso,
            asset=asset,
            weight=float(w),
            asset_return=float(r),
            weighted_return=float(w * r),
            turnover_contribution=float(to),
        ))


def evaluate_dyfo_policy(
    *,
    ckpt: dict,
    data: dict,
    config: DyFOConfig,
    encoder: nn.Module,
    policy: nn.Module,
    episodes: List[List[int]],
    warm_dates: List[int],
    device: torch.device,
    grid_id: str,
    window_idx: int,
    seed: int,
    condition: str,
    regularizer: RiskRegularizer,
) -> Tuple[List[EvalRecord], List[StepRecord]]:
    encoder.eval()
    policy.eval()
    graph = data["graph"]
    edge_index = graph.get_full_edge_index().to(device)
    edge_type_ids = graph.get_edge_type_ids().to(device)
    edge_ts = torch.zeros(edge_index.shape[1], device=device)
    get_nf = _node_feature_getter(data)
    env = PortfolioEpisodeEnv(data["prices"], ckpt["universe"], device)
    records: List[EvalRecord] = []
    step_rows: List[StepRecord] = []

    for ep_idx, ep_dates in enumerate(episodes):
        encoder.reset_state()
        _warm_encoder(encoder, data, warm_dates, edge_index, edge_type_ids, edge_ts, get_nf, device)
        daily, entropies, turnovers = [], [], []
        prev_w = None
        weights_ok = True
        with torch.no_grad():
            for today, tomorrow in zip(ep_dates[:-1], ep_dates[1:]):
                nf = get_nf(today).to(device)
                events = data["events_by_date"].get(today, [])
                t = float(today) + 0.99
                encoder.advance_day(events, nf, edge_index, edge_type_ids, edge_ts, t)
                z = encoder.get_node_embeddings(nf, edge_index, edge_type_ids, edge_ts, t)
                weights = regularizer.transform_weights(policy(z))
                weights_ok &= bool((weights >= -1e-6).all() and abs(weights.sum() - 1.0) < 1e-4)
                next_ret = env.next_return(today, tomorrow)
                daily.append(float(_portfolio_log_return(weights, next_ret)))
                entropies.append(_entropy(weights))
                if prev_w is not None:
                    turnovers.append(float(torch.abs(weights - prev_w).sum()))
                _append_step_rows(
                    step_rows,
                    grid_id=grid_id,
                    window_idx=window_idx,
                    seed=seed,
                    episode_idx=ep_idx,
                    condition=condition,
                    date_iso=_int_day_to_iso(tomorrow),
                    universe=ckpt["universe"],
                    weights=weights,
                    next_ret=next_ret,
                    prev_w=prev_w,
                )
                prev_w = weights.detach()
        records.append(_record_eval(
            grid_id=grid_id,
            window_idx=window_idx,
            seed=seed,
            episode_idx=ep_idx,
            condition=condition,
            ep_dates=ep_dates,
            daily=daily,
            entropies=entropies,
            turnovers=turnovers,
            weights_ok=weights_ok,
            regularizer=regularizer,
        ))
    return records, step_rows


def evaluate_raw_policy(
    *,
    ckpt: dict,
    data: dict,
    policy: nn.Module,
    episodes: List[List[int]],
    device: torch.device,
    grid_id: str,
    window_idx: int,
    seed: int,
    condition: str,
    regularizer: RiskRegularizer,
) -> Tuple[List[EvalRecord], List[StepRecord]]:
    policy.eval()
    env = PortfolioEpisodeEnv(data["prices"], ckpt["universe"], device)
    records: List[EvalRecord] = []
    step_rows: List[StepRecord] = []

    for ep_idx, ep_dates in enumerate(episodes):
        daily, entropies, turnovers = [], [], []
        prev_w = None
        weights_ok = True
        with torch.no_grad():
            for today, tomorrow in zip(ep_dates[:-1], ep_dates[1:]):
                today_ts = pd.Timestamp(_int_day_to_iso(today))
                if today_ts not in env.prices_df.index:
                    continue
                z = _raw_state(env.prices_df, ckpt["universe"], today_ts, window=10, device=device)
                weights = regularizer.transform_weights(policy(z))
                weights_ok &= bool((weights >= -1e-6).all() and abs(weights.sum() - 1.0) < 1e-4)
                next_ret = env.next_return(today, tomorrow)
                daily.append(float(_portfolio_log_return(weights, next_ret)))
                entropies.append(_entropy(weights))
                if prev_w is not None:
                    turnovers.append(float(torch.abs(weights - prev_w).sum()))
                _append_step_rows(
                    step_rows,
                    grid_id=grid_id,
                    window_idx=window_idx,
                    seed=seed,
                    episode_idx=ep_idx,
                    condition=condition,
                    date_iso=_int_day_to_iso(tomorrow),
                    universe=ckpt["universe"],
                    weights=weights,
                    next_ret=next_ret,
                    prev_w=prev_w,
                )
                prev_w = weights.detach()
        records.append(_record_eval(
            grid_id=grid_id,
            window_idx=window_idx,
            seed=seed,
            episode_idx=ep_idx,
            condition=condition,
            ep_dates=ep_dates,
            daily=daily,
            entropies=entropies,
            turnovers=turnovers,
            weights_ok=weights_ok,
            regularizer=regularizer,
        ))
    return records, step_rows


def evaluate_static_baseline(
    *,
    ckpt: dict,
    data: dict,
    episodes: List[List[int]],
    grid_id: str,
    window_idx: int,
    seed: int,
    condition: str,
    alpha: float,
    regularizer: RiskRegularizer,
    device: torch.device,
) -> Tuple[List[EvalRecord], List[StepRecord]]:
    env = PortfolioEpisodeEnv(data["prices"], ckpt["universe"], device)
    n = len(ckpt["universe"])
    records: List[EvalRecord] = []
    step_rows: List[StepRecord] = []

    for ep_idx, ep_dates in enumerate(episodes):
        daily, entropies, turnovers = [], [], []
        prev_w = None
        weights_ok = True
        for today, tomorrow in zip(ep_dates[:-1], ep_dates[1:]):
            today_ts = pd.Timestamp(_int_day_to_iso(today))
            tom_ts = pd.Timestamp(_int_day_to_iso(tomorrow))
            if today_ts not in env.prices_df.index or tom_ts not in env.prices_df.index:
                continue
            if condition == "EWMA-GMVP":
                loc = env.prices_df.index.get_loc(today_ts)
                hist = env.prices_df.iloc[:loc + 1].pct_change().dropna()
                if len(hist) < 5:
                    continue
                weights = _gmvp(_ewma_cov(torch.tensor(hist.values, dtype=torch.float32), alpha=alpha)).to(device)
            elif condition == "EqualWeight":
                weights = torch.full((n,), 1.0 / n, dtype=torch.float32, device=device)
            else:
                raise ValueError(f"Unknown static baseline: {condition}")

            weights_ok &= bool((weights >= -1e-6).all() and abs(weights.sum() - 1.0) < 1e-4)
            next_ret = env.next_return(today, tomorrow)
            daily.append(float(_portfolio_log_return(weights, next_ret)))
            entropies.append(_entropy(weights))
            if prev_w is not None:
                turnovers.append(float(torch.abs(weights - prev_w).sum()))
            _append_step_rows(
                step_rows,
                grid_id=grid_id,
                window_idx=window_idx,
                seed=seed,
                episode_idx=ep_idx,
                condition=condition,
                date_iso=_int_day_to_iso(tomorrow),
                universe=ckpt["universe"],
                weights=weights,
                next_ret=next_ret,
                prev_w=prev_w,
            )
            prev_w = weights.detach()
        records.append(_record_eval(
            grid_id=grid_id,
            window_idx=window_idx,
            seed=seed,
            episode_idx=ep_idx,
            condition=condition,
            ep_dates=ep_dates,
            daily=daily,
            entropies=entropies,
            turnovers=turnovers,
            weights_ok=weights_ok,
            regularizer=regularizer,
        ))
    return records, step_rows


# ---------------------------------------------------------------------------
# 6. Subperiod and attribution analysis
# ---------------------------------------------------------------------------


def _records_to_frame(records: List[EvalRecord]) -> pd.DataFrame:
    rows = []
    for rec in records:
        row = asdict(rec)
        row["daily_log_returns"] = json.dumps(row["daily_log_returns"])
        rows.append(row)
    return pd.DataFrame(rows)


def _rows_to_frame(records: List[EvalRecord]) -> pd.DataFrame:
    """Backward-compatible alias used by protocol tests."""
    return _records_to_frame(records)


def _steps_to_frame(steps: List[StepRecord]) -> pd.DataFrame:
    return pd.DataFrame([asdict(x) for x in steps])


def add_regime_labels(episodes_df: pd.DataFrame) -> pd.DataFrame:
    if episodes_df.empty:
        return episodes_df
    df = episodes_df.copy()
    df["regime"] = [
        _episode_regime(start, end, mdd)
        for start, end, mdd in zip(df["test_start"], df["test_end"], df["max_drawdown"])
    ]
    return df


def build_regime_summary(episodes_df: pd.DataFrame) -> pd.DataFrame:
    if episodes_df.empty:
        return pd.DataFrame()
    metrics = [
        "cumulative_log_return",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "cdar",
        "mean_entropy",
        "mean_turnover",
    ]
    return (
        episodes_df.groupby(["grid_id", "regime", "condition"], as_index=False)[metrics]
        .mean()
        .sort_values(["grid_id", "regime", "condition"])
    )


def build_asset_attribution(step_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if step_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = step_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["regime"] = [
        _episode_regime(d.isoformat(), d.isoformat(), None)
        for d in df["date"]
    ]
    df["risk_proxy"] = df["weight"].abs() * df["asset_return"].pow(2)
    attr = (
        df.groupby(["grid_id", "regime", "condition", "asset"], as_index=False)
        .agg(
            mean_weight=("weight", "mean"),
            return_contribution=("weighted_return", "sum"),
            risk_contribution=("risk_proxy", "sum"),
            turnover_contribution=("turnover_contribution", "sum"),
        )
    )
    alloc_ts = (
        df.groupby(["grid_id", "condition", "date", "asset"], as_index=False)
        .agg(weight=("weight", "mean"), weighted_return=("weighted_return", "sum"))
    )
    return attr, alloc_ts


# ---------------------------------------------------------------------------
# 7. Grid search, summaries and bootstrap
# ---------------------------------------------------------------------------


def _condition_summary(df: pd.DataFrame) -> dict:
    out = {}
    for (grid_id, cond), g in df.groupby(["grid_id", "condition"]):
        out.setdefault(grid_id, {})[cond] = {
            "n_episodes": int(len(g)),
            "mean_cum_log_ret": float(g["cumulative_log_return"].mean()),
            "mean_sharpe": float(g["sharpe"].replace([math.inf, -math.inf], math.nan).mean()),
            "mean_sortino": float(g["sortino"].replace([math.inf, -math.inf], math.nan).mean()),
            "mean_calmar": float(g["calmar"].replace([math.inf, -math.inf], math.nan).mean()),
            "mean_cvar_5": float(g["cvar_5"].mean()),
            "mean_realized_volatility": float(g["realized_volatility"].mean()),
            "mean_max_drawdown": float(g["max_drawdown"].mean()),
            "mean_cdar": float(g["cdar"].mean()),
            "mean_entropy": float(g["mean_entropy"].mean()),
            "mean_turnover": float(g["mean_turnover"].mean()),
            "all_weights_valid": bool(g["weights_always_valid"].all()),
        }
    return out


def _paired_summary(df: pd.DataFrame, metric: str, lhs: str, rhs: str, grid_id: Optional[str] = None) -> dict:
    sub = df if grid_id is None else df[df["grid_id"] == grid_id]
    keys = ["window_idx", "seed", "episode_idx"]
    pivot = sub.pivot_table(index=keys, columns="condition", values=metric, aggfunc="mean")
    if lhs not in pivot or rhs not in pivot:
        return {"n": 0}
    diff = (pivot[lhs] - pivot[rhs]).dropna()
    if len(diff) == 0:
        return {"n": 0}
    lower_is_better = {
        "max_drawdown",
        "mean_turnover",
        "cdar",
        "cvar_5",
        "realized_volatility",
    }
    wins = int((diff < 0).sum()) if metric in lower_is_better else int((diff > 0).sum())
    ties = int((diff == 0).sum())
    n_nonzero = int((diff != 0).sum())
    if n_nonzero == 0:
        p_sign = 1.0
    else:
        k = min(wins, n_nonzero - wins)
        cdf = sum(math.comb(n_nonzero, i) for i in range(k + 1)) / (2 ** n_nonzero)
        p_sign = min(1.0, 2.0 * cdf)
    return {
        "n": int(len(diff)),
        "mean_diff": float(diff.mean()),
        "median_diff": float(diff.median()),
        "win_rate": float(wins / len(diff)),
        "wins": wins,
        "ties": ties,
        "sign_test_p": float(p_sign),
    }


def _bootstrap_daily_diff_ci(
    df: pd.DataFrame,
    lhs: str,
    rhs: str,
    n_bootstrap: int,
    block_len: int,
    seed: int,
    grid_id: Optional[str] = None,
) -> dict:
    if n_bootstrap <= 0:
        return {"n_bootstrap": 0}
    sub = df if grid_id is None else df[df["grid_id"] == grid_id]
    keys = ["window_idx", "seed", "episode_idx"]
    pivot = sub.pivot_table(index=keys, columns="condition", values="daily_log_returns", aggfunc="first")
    if lhs not in pivot or rhs not in pivot:
        return {"n_bootstrap": 0}
    diffs: List[float] = []
    for _, row in pivot.dropna(subset=[lhs, rhs]).iterrows():
        a = json.loads(row[lhs])
        b = json.loads(row[rhs])
        diffs.extend([float(x) - float(y) for x, y in zip(a, b)])
    if not diffs:
        return {"n_bootstrap": 0}
    rng = random.Random(seed)
    samples = []
    n = len(diffs)
    block_len = max(1, min(block_len, n))
    for _ in range(n_bootstrap):
        sampled = []
        while len(sampled) < n:
            start = rng.randrange(0, n)
            for j in range(block_len):
                sampled.append(diffs[(start + j) % n])
                if len(sampled) >= n:
                    break
        samples.append(sum(sampled))
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return {
        "n_bootstrap": int(n_bootstrap),
        "n_daily_diffs": int(n),
        "mean_daily_diff": float(sum(diffs) / n),
        "sum_diff_ci95": [float(lo), float(hi)],
    }


def build_paired_payload(df: pd.DataFrame, cfg: ExperimentConfig) -> Tuple[dict, dict]:
    paired = {}
    bootstrap = {}
    comparisons = [
        ("DyFO-DRL", "EWMA-GMVP"),
        ("DyFO-DRL+", "EWMA-GMVP"),
        ("DyFO-DRL", "EqualWeight"),
        ("DyFO-DRL+", "EqualWeight"),
        ("DyFO-DRL+", "Raw-DRL+"),
        ("DyFO-DRL", "Raw-DRL"),
    ]
    metrics = ["cumulative_log_return", "sharpe", "sortino", "calmar", "max_drawdown", "mean_turnover"]
    for grid_id in sorted(df["grid_id"].unique()):
        paired[grid_id] = {}
        bootstrap[grid_id] = {}
        for lhs, rhs in comparisons:
            name = f"{lhs} vs {rhs}"
            paired[grid_id][name] = {
                metric: _paired_summary(df, metric, lhs, rhs, grid_id)
                for metric in metrics
            }
            bootstrap[grid_id][name] = _bootstrap_daily_diff_ci(
                df,
                lhs=lhs,
                rhs=rhs,
                grid_id=grid_id,
                n_bootstrap=cfg.n_bootstrap,
                block_len=cfg.bootstrap_block_len,
                seed=cfg.bootstrap_seed,
            )
    return paired, bootstrap


def build_grid_sensitivity(episodes_df: pd.DataFrame) -> pd.DataFrame:
    if episodes_df.empty:
        return pd.DataFrame()
    rows = []
    for grid_id, g in episodes_df.groupby("grid_id"):
        pivot = g.pivot_table(
            index=["window_idx", "seed", "episode_idx"],
            columns="condition",
            values="cumulative_log_return",
            aggfunc="mean",
        )
        dyfo = pivot["DyFO-DRL+"] if "DyFO-DRL+" in pivot else None
        ewma = pivot["EWMA-GMVP"] if "EWMA-GMVP" in pivot else None
        equal = pivot["EqualWeight"] if "EqualWeight" in pivot else None
        rows.append({
            "grid_id": grid_id,
            "dyfo_plus_mean_cumret": float(dyfo.mean()) if dyfo is not None else float("nan"),
            "dyfo_plus_vs_ewma_mean_diff": float((dyfo - ewma).dropna().mean()) if dyfo is not None and ewma is not None else float("nan"),
            "dyfo_plus_vs_equal_mean_diff": float((dyfo - equal).dropna().mean()) if dyfo is not None and equal is not None else float("nan"),
            "dyfo_plus_mean_sharpe": float(g[g["condition"] == "DyFO-DRL+"]["sharpe"].mean()),
            "dyfo_plus_mean_turnover": float(g[g["condition"] == "DyFO-DRL+"]["mean_turnover"].mean()),
            "dyfo_plus_mean_mdd": float(g[g["condition"] == "DyFO-DRL+"]["max_drawdown"].mean()),
        })
    return pd.DataFrame(rows).sort_values("dyfo_plus_vs_ewma_mean_diff", ascending=False)


# ---------------------------------------------------------------------------
# 8. Plots and reports
# ---------------------------------------------------------------------------


def _safe_plot_dir(out_dir: Path) -> Path:
    path = out_dir / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_plots(out_dir: Path, episodes_df: pd.DataFrame, step_df: pd.DataFrame, attr_df: pd.DataFrame) -> None:
    if plt is None or episodes_df.empty:
        return
    plot_dir = _safe_plot_dir(out_dir)
    for grid_id, g in episodes_df.groupby("grid_id"):
        fig, ax = plt.subplots(figsize=(11, 5))
        for cond, cdf in g.groupby("condition"):
            daily = []
            dates = []
            for _, row in cdf.sort_values(["window_idx", "seed", "episode_idx"]).iterrows():
                vals = json.loads(row["daily_log_returns"])
                start = pd.Timestamp(row["test_start"])
                for i, v in enumerate(vals):
                    daily.append(float(v))
                    dates.append(start + pd.Timedelta(days=i))
            if not daily:
                continue
            curve = pd.Series(daily, index=dates).groupby(level=0).mean().cumsum().apply(math.exp)
            ax.plot(curve.index, curve.values, label=cond)
        ax.set_title(f"Cumulative performance - {grid_id}")
        ax.set_ylabel("Growth of $1")
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{grid_id}_cumulative_performance.png", dpi=180)
        plt.close(fig)

    if not step_df.empty:
        for grid_id, g in step_df.groupby("grid_id"):
            dyfo = g[g["condition"] == "DyFO-DRL+"]
            if dyfo.empty:
                continue
            alloc = dyfo.groupby(["date", "asset"], as_index=False)["weight"].mean()
            top_assets = (
                alloc.groupby("asset")["weight"].mean().sort_values(ascending=False).head(8).index.tolist()
            )
            alloc = alloc[alloc["asset"].isin(top_assets)].pivot(index="date", columns="asset", values="weight").fillna(0.0)
            fig, ax = plt.subplots(figsize=(12, 5))
            alloc.plot.area(ax=ax, linewidth=0)
            ax.set_title(f"DyFO-DRL+ allocation over time - {grid_id}")
            ax.set_ylabel("Weight")
            fig.tight_layout()
            fig.savefig(plot_dir / f"{grid_id}_dyfo_plus_allocation.png", dpi=180)
            plt.close(fig)

    if not attr_df.empty:
        for grid_id, g in attr_df.groupby("grid_id"):
            dyfo = g[g["condition"] == "DyFO-DRL+"]
            if dyfo.empty:
                continue
            heat = dyfo.pivot_table(index="regime", columns="asset", values="return_contribution", aggfunc="sum").fillna(0.0)
            if heat.empty:
                continue
            fig, ax = plt.subplots(figsize=(12, 5))
            if sns is not None:
                sns.heatmap(heat, cmap="RdYlGn", center=0.0, ax=ax)
            else:
                im = ax.imshow(heat.values, aspect="auto", cmap="RdYlGn")
                ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=90)
                ax.set_yticks(range(len(heat.index)), heat.index)
                fig.colorbar(im, ax=ax)
            ax.set_title(f"DyFO-DRL+ return contribution by regime - {grid_id}")
            fig.tight_layout()
            fig.savefig(plot_dir / f"{grid_id}_dyfo_plus_regime_contribution_heatmap.png", dpi=180)
            plt.close(fig)


def write_report(out_dir: Path, payload: dict) -> None:
    lines = [
        "# DyFO Risk-Aware DRL Walk-Forward Report",
        "",
        "## Protocol",
        f"- checkpoint_mode: `{payload['protocol']['checkpoint_mode']}`",
        f"- causal_tgat: `{payload['protocol']['causal_tgat']}`",
        f"- windows: `{payload['protocol']['n_windows']}`",
        f"- seeds: `{payload['protocol']['seeds']}`",
        f"- train/val/test days: `{payload['protocol']['train_days']}` / "
        f"`{payload['protocol']['val_days']}` / `{payload['protocol']['test_days']}`",
        f"- episode_len: `{payload['protocol']['episode_len']}`",
        f"- drl_episodes: `{payload['protocol']['drl_episodes']}`",
        f"- raw_optimizer: `{payload['protocol'].get('raw_optimizer', 'direct')}`",
        "",
        "## Hyperparameter Sensitivity",
        "| Grid | DyFO+ CumRet | DyFO+ vs EWMA | DyFO+ vs EqualWeight | DyFO+ Sharpe | DyFO+ Turnover | DyFO+ MDD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("grid_sensitivity", []):
        lines.append(
            f"| {row['grid_id']} | {row['dyfo_plus_mean_cumret']:+.4f} | "
            f"{row['dyfo_plus_vs_ewma_mean_diff']:+.4f} | "
            f"{row['dyfo_plus_vs_equal_mean_diff']:+.4f} | "
            f"{row['dyfo_plus_mean_sharpe']:.3f} | "
            f"{row['dyfo_plus_mean_turnover']:.3f} | "
            f"{row['dyfo_plus_mean_mdd']:.3f} |"
        )

    for grid_id, summary in payload["condition_summary"].items():
        lines.extend([
            "",
            f"## Condition Means - `{grid_id}`",
            "| Condition | N | CumRet | Sharpe | Sortino | Calmar | MDD | CDaR | Entropy | Turnover |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for cond in CONDITIONS:
            s = summary.get(cond)
            if not s:
                continue
            lines.append(
                f"| {cond} | {s['n_episodes']} | {s['mean_cum_log_ret']:+.4f} | "
                f"{s['mean_sharpe']:.3f} | {s['mean_sortino']:.3f} | "
                f"{s['mean_calmar']:.3f} | {s['mean_max_drawdown']:.3f} | "
                f"{s['mean_cdar']:.3f} | {s['mean_entropy']:.3f} | "
                f"{s['mean_turnover']:.3f} |"
            )

    lines.extend([
        "",
        "## Paired Evidence",
        "| Grid | Comparison | Metric | N | Mean Diff | Median Diff | Win Rate | Sign p |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for grid_id, comps in payload["paired"].items():
        for name, item in comps.items():
            for metric, stats in item.items():
                if stats.get("n", 0) == 0:
                    continue
                lines.append(
                    f"| {grid_id} | {name} | {metric} | {stats['n']} | "
                    f"{stats['mean_diff']:+.4f} | {stats['median_diff']:+.4f} | "
                    f"{100 * stats['win_rate']:.1f}% | {stats['sign_test_p']:.4f} |"
                )

    lines.extend([
        "",
        "## Regime Summary",
        "| Grid | Regime | Condition | CumRet | Sharpe | MDD | Turnover |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in payload.get("regime_summary", [])[:300]:
        lines.append(
            f"| {row['grid_id']} | {row['regime']} | {row['condition']} | "
            f"{row['cumulative_log_return']:+.4f} | {row['sharpe']:.3f} | "
            f"{row['max_drawdown']:.3f} | {row['mean_turnover']:.3f} |"
        )

    collapse_notes = []
    for grid_id, summary in payload["condition_summary"].items():
        raw = summary.get("Raw-DRL")
        equal = summary.get("EqualWeight")
        if not raw or not equal:
            continue
        entropy_diff = abs(raw["mean_entropy"] - equal["mean_entropy"])
        cumret_diff = abs(raw["mean_cum_log_ret"] - equal["mean_cum_log_ret"])
        if entropy_diff <= 0.005 and cumret_diff <= 0.001:
            collapse_notes.append((grid_id, raw, equal, entropy_diff, cumret_diff))

    if collapse_notes:
        lines.extend([
            "",
            "## Raw-DRL Diagnostic",
            "Raw-DRL is effectively indistinguishable from EqualWeight in the "
            "following grid(s). Treat Raw-DRL as a collapse diagnostic rather "
            "than a converged raw-feature ablation in this regime.",
            "",
            "| Grid | Raw CumRet | EqualWeight CumRet | Raw Entropy | EqualWeight Entropy |",
            "|---|---:|---:|---:|---:|",
        ])
        for grid_id, raw, equal, _entropy_diff, _cumret_diff in collapse_notes:
            lines.append(
                f"| {grid_id} | {raw['mean_cum_log_ret']:+.4f} | "
                f"{equal['mean_cum_log_ret']:+.4f} | "
                f"{raw['mean_entropy']:.3f} | {equal['mean_entropy']:.3f} |"
            )

    if not payload["protocol"]["causal_tgat"]:
        lines.extend([
            "",
            "## Causality Note",
            "This run reused a pre-existing TGAT checkpoint. Use "
            "`--checkpoint_mode causal` for the official no-leakage protocol.",
        ])

    (out_dir / "dyfo_drl_walkforward_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 9. Main experiment orchestration
# ---------------------------------------------------------------------------


def _load_yaml_config(path: Optional[str]) -> dict:
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is not installed; install pyyaml or use CLI arguments only.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_update_dataclass(cfg: ExperimentConfig, data: dict) -> ExperimentConfig:
    cfg_dict = asdict(cfg)
    for key, value in data.items():
        if key == "risk" and isinstance(value, dict):
            cfg_dict["risk"].update(value)
        elif key in cfg_dict:
            cfg_dict[key] = value
    cfg_dict["risk"] = RiskRegularizerConfig(**cfg_dict["risk"])
    cfg_dict["seeds"] = _parse_seeds(cfg_dict["seeds"])
    cfg_dict["turnover_grid"] = _parse_float_grid(cfg_dict["turnover_grid"])
    cfg_dict["risk_aversion_grid"] = _parse_float_grid(cfg_dict["risk_aversion_grid"])
    cfg_dict["drawdown_penalty_grid"] = _parse_float_grid(cfg_dict["drawdown_penalty_grid"])
    return ExperimentConfig(**cfg_dict)


def _configure_logging(out_dir: Path, level: str) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("dyfo.drl_walkforward")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def _grid_configs(cfg: ExperimentConfig) -> List[Tuple[str, RiskRegularizerConfig]]:
    combos = list(itertools.product(cfg.turnover_grid, cfg.risk_aversion_grid, cfg.drawdown_penalty_grid))
    if cfg.max_grid_runs is not None:
        combos = combos[: cfg.max_grid_runs]
    out = []
    for turnover, risk, drawdown in combos:
        risk_cfg = replace(
            cfg.risk,
            turnover_penalty=turnover,
            risk_aversion=risk,
            drawdown_penalty_weight=drawdown,
        )
        out.append((_grid_id(turnover, risk, drawdown), risk_cfg))
    return out


def run_one_grid(
    *,
    cfg: ExperimentConfig,
    grid_id: str,
    risk_cfg: RiskRegularizerConfig,
    data: dict,
    windows: List[WindowSpec],
    device: torch.device,
    config: DyFOConfig,
    out_dir: Path,
    logger: logging.Logger,
) -> Tuple[List[EvalRecord], List[StepRecord]]:
    grid_dir = out_dir / "experiments" / grid_id
    grid_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if cfg.enable_tensorboard and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(grid_dir / "tensorboard"))
    regularizer = RiskRegularizer(risk_cfg)

    all_records: List[EvalRecord] = []
    all_steps: List[StepRecord] = []

    for window in windows:
        test_episodes = build_episodes(
            window.test_dates,
            episode_len=cfg.episode_len,
            step=cfg.test_episode_step or cfg.episode_len,
        )
        train_policy_dates = window.train_dates + window.val_dates
        train_episodes = build_episodes(
            train_policy_dates,
            episode_len=cfg.episode_len,
            step=cfg.train_episode_step or max(1, cfg.episode_len // 2),
        )
        if not test_episodes or not train_episodes:
            logger.warning("Skipping window=%s because episodes are unavailable", window.window_idx)
            continue

        logger.info(
            "Grid=%s window=%s train=%s..%s val=%s..%s test=%s..%s train_eps=%s test_eps=%s",
            grid_id,
            window.window_idx,
            window.train_start,
            window.train_end,
            window.val_start,
            window.val_end,
            window.test_start,
            window.test_end,
            len(train_episodes),
            len(test_episodes),
        )

        for seed in cfg.seeds:
            ckpt = _prepare_checkpoint(
                cfg=cfg,
                data=data,
                config=config,
                window=window,
                seed=seed,
                device=device,
                out_dir=out_dir,
                logger=logger,
            )
            warm_dates = window.val_dates[-cfg.warmup_days:] if cfg.warmup_days > 0 else []
            train_pool = _cycle_episodes(train_episodes, cfg.drl_episodes)

            logger.info("Training policies | grid=%s window=%s seed=%s", grid_id, window.window_idx, seed)
            dyfo_encoder, dyfo_policy = train_dyfo_policy(
                ckpt=ckpt,
                data=data,
                config=config,
                cfg=cfg,
                episodes=train_pool,
                warm_dates=warm_dates,
                device=device,
                seed=seed,
                regularizer=regularizer,
                improved=False,
                label="DyFO-DRL",
                writer=writer,
            )
            dyfo_plus_encoder, dyfo_plus_policy = train_dyfo_policy(
                ckpt=ckpt,
                data=data,
                config=config,
                cfg=cfg,
                episodes=train_pool,
                warm_dates=warm_dates,
                device=device,
                seed=seed,
                regularizer=regularizer,
                improved=True,
                label="DyFO-DRL+",
                writer=writer,
            )
            raw_policy = train_raw_policy(
                ckpt=ckpt,
                data=data,
                cfg=cfg,
                episodes=train_pool,
                device=device,
                seed=seed,
                regularizer=regularizer,
                improved=False,
                label="Raw-DRL",
                writer=writer,
            )
            raw_plus_policy = train_raw_policy(
                ckpt=ckpt,
                data=data,
                cfg=cfg,
                episodes=train_pool,
                device=device,
                seed=seed,
                regularizer=regularizer,
                improved=True,
                label="Raw-DRL+",
                writer=writer,
            )

            if cfg.save_models:
                torch.save(
                    {
                        "dyfo_policy": dyfo_policy.state_dict(),
                        "dyfo_plus_policy": dyfo_plus_policy.state_dict(),
                        "raw_policy": raw_policy.state_dict(),
                        "raw_plus_policy": raw_plus_policy.state_dict(),
                        "risk_config": asdict(risk_cfg),
                        "window": asdict(window),
                        "seed": seed,
                    },
                    grid_dir / f"policies_window_{window.window_idx:02d}_seed_{seed}.pt",
                )

            logger.info("Evaluating OOS | grid=%s window=%s seed=%s", grid_id, window.window_idx, seed)
            for recs, steps in [
                evaluate_dyfo_policy(
                    ckpt=ckpt,
                    data=data,
                    config=config,
                    encoder=dyfo_encoder,
                    policy=dyfo_policy,
                    episodes=test_episodes,
                    warm_dates=warm_dates,
                    device=device,
                    grid_id=grid_id,
                    window_idx=window.window_idx,
                    seed=seed,
                    condition="DyFO-DRL",
                    regularizer=regularizer,
                ),
                evaluate_dyfo_policy(
                    ckpt=ckpt,
                    data=data,
                    config=config,
                    encoder=dyfo_plus_encoder,
                    policy=dyfo_plus_policy,
                    episodes=test_episodes,
                    warm_dates=warm_dates,
                    device=device,
                    grid_id=grid_id,
                    window_idx=window.window_idx,
                    seed=seed,
                    condition="DyFO-DRL+",
                    regularizer=regularizer,
                ),
                evaluate_raw_policy(
                    ckpt=ckpt,
                    data=data,
                    policy=raw_policy,
                    episodes=test_episodes,
                    device=device,
                    grid_id=grid_id,
                    window_idx=window.window_idx,
                    seed=seed,
                    condition="Raw-DRL",
                    regularizer=regularizer,
                ),
                evaluate_raw_policy(
                    ckpt=ckpt,
                    data=data,
                    policy=raw_plus_policy,
                    episodes=test_episodes,
                    device=device,
                    grid_id=grid_id,
                    window_idx=window.window_idx,
                    seed=seed,
                    condition="Raw-DRL+",
                    regularizer=regularizer,
                ),
                evaluate_static_baseline(
                    ckpt=ckpt,
                    data=data,
                    episodes=test_episodes,
                    grid_id=grid_id,
                    window_idx=window.window_idx,
                    seed=seed,
                    condition="EWMA-GMVP",
                    alpha=cfg.alpha_ewma,
                    regularizer=regularizer,
                    device=device,
                ),
                evaluate_static_baseline(
                    ckpt=ckpt,
                    data=data,
                    episodes=test_episodes,
                    grid_id=grid_id,
                    window_idx=window.window_idx,
                    seed=seed,
                    condition="EqualWeight",
                    alpha=cfg.alpha_ewma,
                    regularizer=regularizer,
                    device=device,
                ),
            ]:
                all_records.extend(recs)
                all_steps.extend(steps)

    if writer is not None:
        writer.close()
    return all_records, all_steps


def run(cfg: ExperimentConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    logger = _configure_logging(out_dir, cfg.log_level)
    if cfg.raw_optimizer not in {"direct", "ppo"}:
        raise ValueError(f"Unsupported raw_optimizer: {cfg.raw_optimizer}")
    if cfg.raw_ppo_clip <= 0:
        raise ValueError("raw_ppo_clip must be positive")
    if cfg.raw_ppo_epochs < 1:
        raise ValueError("raw_ppo_epochs must be >= 1")
    if cfg.raw_ppo_concentration <= 0:
        raise ValueError("raw_ppo_concentration must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.cpu else "cpu")
    setup_logging("dyfo.portfolio_walkforward_data", log_to_file=False)
    dyfo_config = DyFOConfig(model_variant="tgat")
    data_config = DataConfig(
        tickers=UNIVERSE,
        benchmark_ticker="SPY",
        start_date=cfg.start,
        end_date=cfg.end,
    )

    logger.info("Universe (%s assets): %s", len(UNIVERSE), UNIVERSE)
    logger.info("Loading/preparing data %s..%s", cfg.start, cfg.end)
    data = load_or_prepare_data(
        tickers=UNIVERSE,
        start=cfg.start,
        end=cfg.end,
        benchmark="SPY",
        config=dyfo_config,
        data_config=data_config,
        logger=logger,
    )

    all_dates = [
        d for d in data["sorted_dates"]
        if dt.date.fromisoformat(cfg.start) <= _date(d) <= dt.date.fromisoformat(cfg.end)
    ]
    windows = _build_walk_forward_windows(
        all_dates=all_dates,
        train_days=cfg.train_days,
        val_days=cfg.val_days,
        test_days=cfg.test_days,
        step_days=cfg.step_days,
        max_windows=cfg.max_windows,
    )
    if not windows:
        raise RuntimeError("No walk-forward windows. Reduce day counts or extend date range.")

    grid_specs = _grid_configs(cfg)
    logger.info("Running %s grid configuration(s) across %s window(s)", len(grid_specs), len(windows))

    all_records: List[EvalRecord] = []
    all_steps: List[StepRecord] = []
    for grid_id, risk_cfg in grid_specs:
        logger.info("Starting grid=%s risk=%s", grid_id, risk_cfg)
        recs, steps = run_one_grid(
            cfg=cfg,
            grid_id=grid_id,
            risk_cfg=risk_cfg,
            data=data,
            windows=windows,
            device=device,
            config=dyfo_config,
            out_dir=out_dir,
            logger=logger,
        )
        all_records.extend(recs)
        all_steps.extend(steps)

    if not all_records:
        raise RuntimeError("No evaluation records were produced.")

    episodes_df = add_regime_labels(_records_to_frame(all_records))
    step_df = _steps_to_frame(all_steps)
    attr_df, alloc_ts_df = build_asset_attribution(step_df)
    regime_df = build_regime_summary(episodes_df)
    sensitivity_df = build_grid_sensitivity(episodes_df)

    episodes_csv = out_dir / "dyfo_drl_walkforward_episodes.csv"
    steps_csv = out_dir / "dyfo_drl_walkforward_daily_asset_attribution.csv"
    attr_csv = out_dir / "dyfo_drl_walkforward_regime_asset_attribution.csv"
    alloc_csv = out_dir / "dyfo_drl_walkforward_allocation_timeseries.csv"
    regime_csv = out_dir / "dyfo_drl_walkforward_regime_summary.csv"
    sensitivity_csv = out_dir / "dyfo_drl_walkforward_grid_sensitivity.csv"
    episodes_df.to_csv(episodes_csv, index=False)
    step_df.to_csv(steps_csv, index=False)
    attr_df.to_csv(attr_csv, index=False)
    alloc_ts_df.to_csv(alloc_csv, index=False)
    regime_df.to_csv(regime_csv, index=False)
    sensitivity_df.to_csv(sensitivity_csv, index=False)

    paired, daily_bootstrap = build_paired_payload(episodes_df, cfg)
    payload = {
        "protocol": {
            "checkpoint_mode": cfg.checkpoint_mode,
            "causal_tgat": cfg.checkpoint_mode == "causal",
            "universe": UNIVERSE,
            "start": cfg.start,
            "end": cfg.end,
            "train_days": cfg.train_days,
            "val_days": cfg.val_days,
            "test_days": cfg.test_days,
            "step_days": cfg.step_days,
            "episode_len": cfg.episode_len,
            "drl_episodes": cfg.drl_episodes,
            "seeds": cfg.seeds,
            "n_windows": len(windows),
            "n_eval_records": int(len(episodes_df)),
            "risk_base_config": asdict(cfg.risk),
            "raw_optimizer": cfg.raw_optimizer,
            "raw_ppo_clip": cfg.raw_ppo_clip,
            "raw_ppo_epochs": cfg.raw_ppo_epochs,
            "raw_ppo_value_coef": cfg.raw_ppo_value_coef,
            "raw_ppo_concentration": cfg.raw_ppo_concentration,
            "grid": {
                "turnover_grid": cfg.turnover_grid,
                "risk_aversion_grid": cfg.risk_aversion_grid,
                "drawdown_penalty_grid": cfg.drawdown_penalty_grid,
            },
        },
        "windows": [
            {
                "window_idx": w.window_idx,
                "train_start": w.train_start,
                "train_end": w.train_end,
                "val_start": w.val_start,
                "val_end": w.val_end,
                "test_start": w.test_start,
                "test_end": w.test_end,
                "n_train_dates": len(w.train_dates),
                "n_val_dates": len(w.val_dates),
                "n_test_dates": len(w.test_dates),
            }
            for w in windows
        ],
        "condition_summary": _condition_summary(episodes_df),
        "paired": paired,
        "daily_bootstrap": daily_bootstrap,
        "regime_summary": regime_df.to_dict(orient="records"),
        "grid_sensitivity": sensitivity_df.to_dict(orient="records"),
        "artifacts": {
            "episodes_csv": str(episodes_csv),
            "daily_asset_attribution_csv": str(steps_csv),
            "regime_asset_attribution_csv": str(attr_csv),
            "allocation_timeseries_csv": str(alloc_csv),
            "regime_summary_csv": str(regime_csv),
            "grid_sensitivity_csv": str(sensitivity_csv),
            "report_md": str(out_dir / "dyfo_drl_walkforward_report.md"),
        },
    }

    summary_path = out_dir / "dyfo_drl_walkforward_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(out_dir, payload)
    if cfg.make_plots:
        make_plots(out_dir, episodes_df, step_df, attr_df)

    logger.info("Episodes CSV -> %s", episodes_csv)
    logger.info("Summary JSON -> %s", summary_path)
    logger.info("Report MD -> %s", out_dir / "dyfo_drl_walkforward_report.md")
    return payload


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Risk-aware walk-forward/bootstrap evidence for DyFO portfolio DRL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Optional YAML config path.")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--train_days", type=int, default=None)
    parser.add_argument("--val_days", type=int, default=None)
    parser.add_argument("--test_days", type=int, default=None)
    parser.add_argument("--step_days", type=int, default=None)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--episode_len", type=int, default=None)
    parser.add_argument("--train_episode_step", type=int, default=None)
    parser.add_argument("--test_episode_step", type=int, default=None)
    parser.add_argument("--drl_episodes", type=int, default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_drl", type=float, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--finetune_encoder", action="store_true")
    parser.add_argument("--warmup_days", type=int, default=None)
    parser.add_argument("--alpha_ewma", type=float, default=None)
    parser.add_argument("--n_bootstrap", type=int, default=None)
    parser.add_argument("--bootstrap_block_len", type=int, default=None)
    parser.add_argument("--bootstrap_seed", type=int, default=None)
    parser.add_argument("--checkpoint_mode", choices=["causal", "reuse"], default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--reward_objective", choices=["log_return", "sharpe", "sortino", "calmar"], default=None)
    parser.add_argument("--turnover_grid", default=None)
    parser.add_argument("--risk_aversion_grid", default=None)
    parser.add_argument("--drawdown_penalty_grid", default=None)
    parser.add_argument("--max_grid_runs", type=int, default=None)
    parser.add_argument("--volatility_target", type=float, default=None)
    parser.add_argument("--volatility_penalty_weight", type=float, default=None)
    parser.add_argument("--cvar_penalty_weight", type=float, default=None)
    parser.add_argument("--cdar_penalty_weight", type=float, default=None)
    parser.add_argument("--max_weight", type=float, default=None)
    parser.add_argument("--max_weight_penalty_weight", type=float, default=None)
    parser.add_argument("--entropy_bonus_weight", type=float, default=None)
    parser.add_argument("--raw_optimizer", choices=["direct", "ppo"], default=None)
    parser.add_argument("--raw_ppo_clip", type=float, default=None)
    parser.add_argument("--raw_ppo_epochs", type=int, default=None)
    parser.add_argument("--raw_ppo_value_coef", type=float, default=None)
    parser.add_argument("--raw_ppo_concentration", type=float, default=None)
    parser.add_argument("--enable_tensorboard", action="store_true")
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--log_level", default=None)
    args = parser.parse_args()

    cfg = _deep_update_dataclass(ExperimentConfig(), _load_yaml_config(args.config))
    updates = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
    for bool_key in ["finetune_encoder", "force_retrain", "cpu", "enable_tensorboard", "save_models"]:
        if getattr(args, bool_key):
            updates[bool_key] = True
    if args.no_plots:
        updates["make_plots"] = False
    risk_updates = {}
    for key in [
        "reward_objective",
        "volatility_target",
        "volatility_penalty_weight",
        "cvar_penalty_weight",
        "cdar_penalty_weight",
        "max_weight",
        "max_weight_penalty_weight",
        "entropy_bonus_weight",
    ]:
        if key in updates:
            risk_updates[key] = updates.pop(key)
    merged = asdict(cfg)
    for key, value in updates.items():
        if key in {"turnover_grid", "risk_aversion_grid", "drawdown_penalty_grid"}:
            merged[key] = _parse_float_grid(value)
        elif key == "seeds":
            merged[key] = _parse_seeds(value)
        elif key in merged:
            merged[key] = value
    merged["risk"].update(risk_updates)
    merged["risk"] = RiskRegularizerConfig(**merged["risk"])
    return ExperimentConfig(**merged)


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
