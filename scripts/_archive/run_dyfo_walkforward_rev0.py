#!/usr/bin/env python3
"""Walk-forward/bootstrap evaluation for DyFO portfolio DRL.

This runner turns ``scripts/train_dyfo_portfolio.py`` from a convergence
experiment into a paired out-of-sample protocol:

1. Build causal train/validation/test windows.
2. Train or reuse a TGAT checkpoint for each window.
3. Train DRL policies only on pre-test episodes.
4. Evaluate every condition on the same unique test episodes without updates.
5. Aggregate paired evidence vs EWMA-GMVP, EqualWeight and raw DRL ablations.

The default ``checkpoint_mode=causal`` is the statistically clean setting, but
it is expensive because it trains TGAT per walk-forward window. Use
``checkpoint_mode=reuse`` for a fast smoke/profiling run with an existing
checkpoint; the report marks that mode as non-causal for TGAT.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
    run_dyfo_drl_episodes,
    run_raw_drl_episodes,
    run_raw_drl_improved,
    save_checkpoint,
    train_dyfo,
)


CONDITIONS = [
    "DyFO-DRL",
    "DyFO-DRL+",
    "Raw-DRL",
    "Raw-DRL+",
    "EWMA-GMVP",
    "EqualWeight",
]


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
class EvalRecord:
    window_idx: int
    seed: int
    episode_idx: int
    condition: str
    test_start: str
    test_end: str
    cumulative_log_return: float
    sharpe: float
    max_drawdown: float
    mean_entropy: float
    mean_turnover: float
    weights_always_valid: bool
    daily_log_returns: List[float]


def _parse_seeds(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.replace(";", ",").split(",") if x.strip()]


def _date(day: int) -> dt.date:
    return dt.date.fromisoformat(_int_day_to_iso(day))


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _max_drawdown(log_returns: List[float]) -> float:
    equity = 1.0
    peak = 1.0
    mdd = 0.0
    for r in log_returns:
        equity *= math.exp(float(r))
        peak = max(peak, equity)
        mdd = min(mdd, equity / peak - 1.0)
    return float(mdd)


def _record(
    *,
    window_idx: int,
    seed: int,
    episode_idx: int,
    condition: str,
    ep_dates: List[int],
    daily: List[float],
    entropies: List[float],
    turnovers: List[float],
    weights_ok: bool,
) -> EvalRecord:
    return EvalRecord(
        window_idx=window_idx,
        seed=seed,
        episode_idx=episode_idx,
        condition=condition,
    test_start=_int_day_to_iso(ep_dates[0]),
    test_end=_int_day_to_iso(ep_dates[-1]),
    cumulative_log_return=float(sum(daily)),
    sharpe=_sharpe(daily),
    max_drawdown=_max_drawdown(daily),
    mean_entropy=_mean(entropies),
    mean_turnover=_mean(turnovers) if turnovers else 0.0,
    weights_always_valid=weights_ok,
    daily_log_returns=[float(x) for x in daily],
)


def _build_walk_forward_windows(
    all_dates: List[int],
    train_days: int,
    val_days: int,
    test_days: int,
    step_days: int,
    max_windows: int | None,
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


def _prepare_checkpoint(
    *,
    args,
    data: dict,
    config: DyFOConfig,
    window: WindowSpec,
    seed: int,
    device: torch.device,
    out_dir: Path,
) -> dict:
    if args.checkpoint_mode == "reuse":
        return load_checkpoint(Path(args.checkpoint))

    ckpt_path = out_dir / "checkpoints" / f"window_{window.window_idx:02d}_seed_{seed}.pt"
if ckpt_path.exists() and not args.force_retrain:
    return load_checkpoint(ckpt_path)

print(
    f"\n[Window {window.window_idx} seed {seed}] Training causal TGAT "
    f"train={window.train_start}..{window.train_end} "
    f"val={window.val_start}..{window.val_end}"
)
result = train_dyfo(
    data=data,
    num_nodes=len(UNIVERSE),
    train_dates=window.train_dates,
    val_dates=window.val_dates,
    num_epochs=args.epochs,
    lr=args.lr,
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


def _eval_dyfo_policy(
    *,
    ckpt: dict,
    data: dict,
    policy: torch.nn.Module,
    episodes: List[List[int]],
    warm_dates: List[int],
    config: DyFOConfig,
    device: torch.device,
    window_idx: int,
    seed: int,
    condition: str,
) -> List[EvalRecord]:
    encoder = build_encoder(config, ckpt["num_nodes"], variant="tgat").to(device)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()
    policy.eval()

    graph = data["graph"]
    edge_index = graph.get_full_edge_index().to(device)
    edge_type_ids = graph.get_edge_type_ids().to(device)
    edge_ts = torch.zeros(edge_index.shape[1], device=device)
    get_nf = _node_feature_getter(data)

    universe = ckpt["universe"]
    prices_df = data["prices"].reindex(columns=universe).ffill()
    rets_df = prices_df.pct_change().fillna(0.0)

    records: List[EvalRecord] = []
    for ep_idx, ep_dates in enumerate(episodes):
        encoder.reset_state()
    with torch.no_grad():
        for d in warm_dates:
            events = data["events_by_date"].get(d, [])
            nf = get_nf(d).to(device)
            t = float(d) + 0.99
            encoder.advance_day(events, nf, edge_index, edge_type_ids, edge_ts, t)

        daily, entropies, turnovers = [], [], []
        prev_w = None
        weights_ok = True
        for today, tomorrow in zip(ep_dates[:-1], ep_dates[1:]):
            events = data["events_by_date"].get(today, [])
            nf = get_nf(today).to(device)
            t = float(today) + 0.99
            encoder.advance_day(events, nf, edge_index, edge_type_ids, edge_ts, t)
            z = encoder.get_node_embeddings(nf, edge_index, edge_type_ids, edge_ts, t)
            weights = policy(z)
            weights_ok &= bool((weights >= -1e-6).all() and abs(weights.sum() - 1.0) < 1e-4)

            tom_ts = pd.Timestamp(_int_day_to_iso(tomorrow))
            next_ret = (
                torch.tensor(rets_df.loc[tom_ts].values, dtype=torch.float32, device=device).nan_to_num(0.0)
                if tom_ts in rets_df.index
                else torch.zeros(len(universe), device=device)
            )
            daily.append(float(_portfolio_log_return(weights, next_ret)))
            entropies.append(_entropy(weights))
            if prev_w is not None:
                turnovers.append(float(torch.abs(weights.detach() - prev_w).sum()))
            prev_w = weights.detach()

    records.append(_record(
        window_idx=window_idx,
        seed=seed,
        episode_idx=ep_idx,
        condition=condition,
        ep_dates=ep_dates,
        daily=daily,
        entropies=entropies,
        turnovers=turnovers,
        weights_ok=weights_ok,
    ))
    return records


def _eval_raw_policy(
    *,
    ckpt: dict,
    data: dict,
    policy: torch.nn.Module,
    episodes: List[List[int]],
    device: torch.device,
    window_idx: int,
    seed: int,
    condition: str,
) -> List[EvalRecord]:
    policy.eval()
    universe = ckpt["universe"]
    prices_df = data["prices"].reindex(columns=universe).ffill()
    rets_df = prices_df.pct_change().fillna(0.0)
    records: List[EvalRecord] = []

    for ep_idx, ep_dates in enumerate(episodes):
        daily, entropies, turnovers = [], [], []
        prev_w = None
        weights_ok = True
        with torch.no_grad():
            for today, tomorrow in zip(ep_dates[:-1], ep_dates[1:]):
                today_ts = pd.Timestamp(_int_day_to_iso(today))
                tom_ts = pd.Timestamp(_int_day_to_iso(tomorrow))
                if today_ts not in prices_df.index:
                    continue
            z = _raw_state(prices_df, universe, today_ts, window=10, device=device)
            weights = policy(z)
            weights_ok &= bool((weights >= -1e-6).all() and abs(weights.sum() - 1.0) < 1e-4)
            next_ret = (
                torch.tensor(rets_df.loc[tom_ts].values, dtype=torch.float32, device=device).nan_to_num(0.0)
                if tom_ts in rets_df.index
                else torch.zeros(len(universe), device=device)
            )
            daily.append(float(_portfolio_log_return(weights, next_ret)))
            entropies.append(_entropy(weights))
            if prev_w is not None:
                turnovers.append(float(torch.abs(weights.detach() - prev_w).sum()))
            prev_w = weights.detach()
-
    records.append(_record(
        window_idx=window_idx,
        seed=seed,
        episode_idx=ep_idx,
        condition=condition,
        ep_dates=ep_dates,
        daily=daily,
        entropies=entropies,
        turnovers=turnovers,
        weights_ok=weights_ok,
    ))
return records
-
-
-def _eval_static_baseline(
*,
ckpt: dict,
data: dict,
episodes: List[List[int]],
window_idx: int,
seed: int,
condition: str,
alpha: float,
-) -> List[EvalRecord]:
universe = ckpt["universe"]
prices_df = data["prices"].reindex(columns=universe).ffill()
n = len(universe)
records: List[EvalRecord] = []
-
for ep_idx, ep_dates in enumerate(episodes):
    daily, entropies, turnovers = [], [], []
    prev_w = None
    weights_ok = True
    for today, tomorrow in zip(ep_dates[:-1], ep_dates[1:]):
        today_ts = pd.Timestamp(_int_day_to_iso(today))
        tom_ts = pd.Timestamp(_int_day_to_iso(tomorrow))
        if today_ts not in prices_df.index or tom_ts not in prices_df.index:
            continue
-
        if condition == "EWMA-GMVP":
            loc = prices_df.index.get_loc(today_ts)
            hist = prices_df.iloc[:loc + 1].pct_change().dropna()
            if len(hist) < 5:
                continue
            weights = _gmvp(_ewma_cov(torch.tensor(hist.values, dtype=torch.float32), alpha=alpha))
        elif condition == "EqualWeight":
            weights = torch.full((n,), 1.0 / n)
        else:
            raise ValueError(f"Unknown static condition: {condition}")
-
        weights_ok &= bool((weights >= -1e-6).all() and abs(weights.sum() - 1.0) < 1e-4)
        next_ret = torch.tensor(
            prices_df.loc[tom_ts].values / prices_df.loc[today_ts].values - 1.0,
            dtype=torch.float32,
        ).nan_to_num(0.0)
        daily.append(float(_portfolio_log_return(weights, next_ret)))
        entropies.append(_entropy(weights))
        if prev_w is not None:
            turnovers.append(float(torch.abs(weights.detach() - prev_w).sum()))
        prev_w = weights.detach()
-
    records.append(_record(
        window_idx=window_idx,
        seed=seed,
        episode_idx=ep_idx,
        condition=condition,
        ep_dates=ep_dates,
        daily=daily,
        entropies=entropies,
        turnovers=turnovers,
        weights_ok=weights_ok,
    ))
return records
-
-
-def _rows_to_frame(records: List[EvalRecord]) -> pd.DataFrame:
rows = []
for rec in records:
    row = asdict(rec)
    row["daily_log_returns"] = json.dumps(row["daily_log_returns"])
    rows.append(row)
return pd.DataFrame(rows)
-
-
-def _condition_summary(df: pd.DataFrame) -> dict:
out = {}
for cond, g in df.groupby("condition"):
    out[cond] = {
        "n_episodes": int(len(g)),
        "mean_cum_log_ret": float(g["cumulative_log_return"].mean()),
        "mean_sharpe": float(g["sharpe"].replace([math.inf, -math.inf], math.nan).mean()),
        "mean_max_drawdown": float(g["max_drawdown"].mean()),
        "mean_entropy": float(g["mean_entropy"].mean()),
        "mean_turnover": float(g["mean_turnover"].mean()),
        "all_weights_valid": bool(g["weights_always_valid"].all()),
    }
return out
-
-
-def _paired_summary(df: pd.DataFrame, metric: str, lhs: str, rhs: str) -> dict:
keys = ["window_idx", "seed", "episode_idx"]
pivot = df.pivot_table(index=keys, columns="condition", values=metric, aggfunc="mean")
if lhs not in pivot or rhs not in pivot:
    return {"n": 0}
diff = (pivot[lhs] - pivot[rhs]).dropna()
if len(diff) == 0:
    return {"n": 0}
wins = int((diff > 0).sum())
ties = int((diff == 0).sum())
n_nonzero = int((diff != 0).sum())
# Exact two-sided sign-test p-value for non-zero paired differences.
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
-
-
-def _bootstrap_daily_diff_ci(
df: pd.DataFrame,
lhs: str,
rhs: str,
n_bootstrap: int,
block_len: int,
seed: int,
-) -> dict:
if n_bootstrap <= 0:
    return {"n_bootstrap": 0}
-
keys = ["window_idx", "seed", "episode_idx"]
pivot = df.pivot_table(index=keys, columns="condition", values="daily_log_returns", aggfunc="first")
if lhs not in pivot or rhs not in pivot:
    return {"n_bootstrap": 0}
-
diffs: List[float] = []
for _, row in pivot.dropna(subset=[lhs, rhs]).iterrows():
    a = json.loads(row[lhs])
    b = json.loads(row[rhs])
    diffs.extend([float(x) - float(y) for x, y in zip(a, b)])
if not diffs:
    return {"n_bootstrap": 0}
-
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
-
-
-def _write_report(out_dir: Path, payload: dict) -> None:
lines = [
    "# DyFO DRL Walk-Forward Report",
    "",
    "## Protocol",
    f"- checkpoint_mode: `{payload['protocol']['checkpoint_mode']}`",
    f"- causal_tgat: `{payload['protocol']['causal_tgat']}`",
    f"- windows: `{payload['protocol']['n_windows']}`",
    f"- seeds: `{payload['protocol']['seeds']}`",
    f"- train/val/test days: `{payload['protocol']['train_days']}` / "
    f"`{payload['protocol']['val_days']}` / `{payload['protocol']['test_days']}`",
    f"- episode_len: `{payload['protocol']['episode_len']}`",
    "",
    "## Condition Means",
    "| Condition | N | Mean CumRet | Mean Sharpe | Mean MDD | Entropy | Turnover | Weights OK |",
    "|---|---:|---:|---:|---:|---:|---:|---|",
]
for cond in CONDITIONS:
    s = payload["condition_summary"].get(cond)
    if not s:
        continue
    lines.append(
        f"| {cond} | {s['n_episodes']} | {s['mean_cum_log_ret']:+.4f} | "
        f"{s['mean_sharpe']:.3f} | {s['mean_max_drawdown']:.3f} | "
        f"{s['mean_entropy']:.3f} | {s['mean_turnover']:.3f} | "
        f"{s['all_weights_valid']} |"
    )
-
lines.extend([
    "",
    "## Paired Evidence",
    "| Comparison | Metric | N | Mean Diff | Median Diff | Win Rate | Sign p |",
    "|---|---|---:|---:|---:|---:|---:|",
])
for name, item in payload["paired"].items():
    for metric, stats in item.items():
        if stats.get("n", 0) == 0:
            continue
        lines.append(
            f"| {name} | {metric} | {stats['n']} | {stats['mean_diff']:+.4f} | "
            f"{stats['median_diff']:+.4f} | {100 * stats['win_rate']:.1f}% | "
            f"{stats['sign_test_p']:.4f} |"
        )
-
lines.extend([
    "",
    "## Daily Block Bootstrap",
    "| Comparison | Daily Mean Diff | 95% CI For Sum Diff | N Daily Diffs |",
    "|---|---:|---:|---:|",
])
for name, stats in payload["daily_bootstrap"].items():
    if stats.get("n_bootstrap", 0) == 0:
        continue
    lo, hi = stats["sum_diff_ci95"]
    lines.append(
        f"| {name} | {stats['mean_daily_diff']:+.6f} | "
        f"[{lo:+.4f}, {hi:+.4f}] | {stats['n_daily_diffs']} |"
    )
-
if not payload["protocol"]["causal_tgat"]:
    lines.extend([
        "",
        "## Causality Note",
        "This run reused a pre-existing TGAT checkpoint. Use "
        "`--checkpoint_mode causal` for the official no-leakage protocol.",
    ])
-
report_path = out_dir / "dyfo_drl_walkforward_report.md"
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
-
-
-def run(args) -> dict:
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
-
seeds = _parse_seeds(args.seeds)
device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
logger = setup_logging("dyfo.drl_walkforward", log_to_file=False)
config = DyFOConfig(model_variant="tgat")
data_config = DataConfig(
    tickers=UNIVERSE,
    benchmark_ticker="SPY",
    start_date=args.start,
    end_date=args.end,
)
-
print(f"Universe ({len(UNIVERSE)} assets): {UNIVERSE}")
print(f"Loading / preparing market data for {args.start}..{args.end}")
data = load_or_prepare_data(
    tickers=UNIVERSE,
    start=args.start,
    end=args.end,
    benchmark="SPY",
    config=config,
    data_config=data_config,
    logger=logger,
)
-
all_dates = [
    d for d in data["sorted_dates"]
    if dt.date.fromisoformat(args.start) <= _date(d) <= dt.date.fromisoformat(args.end)
]
windows = _build_walk_forward_windows(
    all_dates=all_dates,
    train_days=args.train_days,
    val_days=args.val_days,
    test_days=args.test_days,
    step_days=args.step_days,
    max_windows=args.max_windows,
)
if not windows:
    raise RuntimeError("No walk-forward windows. Reduce day counts or extend date range.")
-
all_records: List[EvalRecord] = []
for window in windows:
    test_episodes = build_episodes(
        window.test_dates,
        episode_len=args.episode_len,
        step=args.test_episode_step or args.episode_len,
    )
    train_policy_dates = window.train_dates + window.val_dates
    train_episodes = build_episodes(
        train_policy_dates,
        episode_len=args.episode_len,
        step=args.train_episode_step or max(1, args.episode_len // 2),
    )
    if not test_episodes:
        print(f"[Window {window.window_idx}] skipped: no full test episodes")
        continue
    if not train_episodes:
        print(f"[Window {window.window_idx}] skipped: no full train episodes")
        continue
-
    print(
        f"\n=== Window {window.window_idx} | "
        f"train {window.train_start}..{window.train_end} | "
        f"val {window.val_start}..{window.val_end} | "
        f"test {window.test_start}..{window.test_end} | "
        f"train_eps={len(train_episodes)} test_eps={len(test_episodes)} ==="
    )
-
    for seed in seeds:
        ckpt = _prepare_checkpoint(
            args=args,
            data=data,
            config=config,
            window=window,
            seed=seed,
            device=device,
            out_dir=out_dir,
        )
        warm_dates = window.val_dates[-args.warmup_days:] if args.warmup_days > 0 else []
        train_pool = _cycle_episodes(train_episodes, args.drl_episodes)
-
        print(f"\n[Window {window.window_idx} seed {seed}] Training DRL policies")
        dyfo_results, dyfo_policy = run_dyfo_drl_episodes(
            ckpt=ckpt,
            data=data,
            episode_dates_list=train_pool,
            warm_dates=warm_dates,
            config=config,
            device=device,
            n_drl_epochs=len(train_pool),
            lr_drl=args.lr_drl,
            seed=seed,
            use_attention_policy=False,
            use_sharpe_reward=False,
            label="DyFO-DRL",
        )
        dyfo_plus_results, dyfo_plus_policy = run_dyfo_drl_episodes(
            ckpt=ckpt,
            data=data,
            episode_dates_list=train_pool,
            warm_dates=warm_dates,
            config=config,
            device=device,
            n_drl_epochs=len(train_pool),
            lr_drl=args.lr_drl,
            seed=seed,
            use_attention_policy=True,
            use_sharpe_reward=True,
            finetune_encoder=args.finetune_encoder,
            n_heads=args.n_heads,
            label="DyFO-DRL+",
        )
        raw_results, raw_policy = run_raw_drl_episodes(
            ckpt=ckpt,
            data=data,
            episode_dates_list=train_pool,
            device=device,
            n_drl_epochs=len(train_pool),
            lr_drl=args.lr_drl,
            seed=seed,
        )
        raw_plus_results, raw_plus_policy = run_raw_drl_improved(
            ckpt=ckpt,
            data=data,
            episode_dates_list=train_pool,
            device=device,
            n_drl_epochs=len(train_pool),
            lr_drl=args.lr_drl,
            seed=seed,
            n_heads=args.n_heads,
            label="Raw-DRL+",
        )
        _ = (dyfo_results, dyfo_plus_results, raw_results, raw_plus_results)
-
        print(f"[Window {window.window_idx} seed {seed}] Evaluating OOS episodes")
        all_records.extend(_eval_dyfo_policy(
            ckpt=ckpt,
            data=data,
            policy=dyfo_policy,
            episodes=test_episodes,
            warm_dates=warm_dates,
            config=config,
            device=device,
            window_idx=window.window_idx,
            seed=seed,
            condition="DyFO-DRL",
        ))
        all_records.extend(_eval_dyfo_policy(
            ckpt=ckpt,
            data=data,
            policy=dyfo_plus_policy,
            episodes=test_episodes,
            warm_dates=warm_dates,
            config=config,
            device=device,
            window_idx=window.window_idx,
            seed=seed,
            condition="DyFO-DRL+",
        ))
        all_records.extend(_eval_raw_policy(
            ckpt=ckpt,
            data=data,
            policy=raw_policy,
            episodes=test_episodes,
            device=device,
            window_idx=window.window_idx,
            seed=seed,
            condition="Raw-DRL",
        ))
        all_records.extend(_eval_raw_policy(
            ckpt=ckpt,
            data=data,
            policy=raw_plus_policy,
            episodes=test_episodes,
            device=device,
            window_idx=window.window_idx,
            seed=seed,
            condition="Raw-DRL+",
        ))
        all_records.extend(_eval_static_baseline(
            ckpt=ckpt,
            data=data,
            episodes=test_episodes,
            window_idx=window.window_idx,
            seed=seed,
            condition="EWMA-GMVP",
            alpha=args.alpha_ewma,
        ))
        all_records.extend(_eval_static_baseline(
            ckpt=ckpt,
            data=data,
            episodes=test_episodes,
            window_idx=window.window_idx,
            seed=seed,
            condition="EqualWeight",
            alpha=args.alpha_ewma,
        ))

if not all_records:
    raise RuntimeError("No evaluation records were produced.")

df = _rows_to_frame(all_records)
csv_path = out_dir / "dyfo_drl_walkforward_episodes.csv"
df.to_csv(csv_path, index=False)

paired = {}
for lhs, rhs in [
    ("DyFO-DRL", "EWMA-GMVP"),
    ("DyFO-DRL+", "EWMA-GMVP"),
    ("DyFO-DRL", "EqualWeight"),
    ("DyFO-DRL+", "EqualWeight"),
    ("DyFO-DRL+", "Raw-DRL+"),
    ("DyFO-DRL", "Raw-DRL"),
]:
    paired[f"{lhs} vs {rhs}"] = {
        metric: _paired_summary(df, metric, lhs, rhs)
        for metric in ["cumulative_log_return", "sharpe", "max_drawdown"]
    }

daily_bootstrap = {
    name: _bootstrap_daily_diff_ci(
        df,
        lhs=name.split(" vs ")[0],
        rhs=name.split(" vs ")[1],
        n_bootstrap=args.n_bootstrap,
        block_len=args.bootstrap_block_len,
        seed=args.bootstrap_seed,
    )
    for name in paired
}

payload = {
    "protocol": {
        "checkpoint_mode": args.checkpoint_mode,
        "causal_tgat": args.checkpoint_mode == "causal",
        "universe": UNIVERSE,
        "start": args.start,
        "end": args.end,
        "train_days": args.train_days,
        "val_days": args.val_days,
        "test_days": args.test_days,
        "step_days": args.step_days,
        "episode_len": args.episode_len,
        "drl_episodes": args.drl_episodes,
        "seeds": seeds,
        "n_windows": len(windows),
        "n_eval_records": int(len(df)),
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
    "condition_summary": _condition_summary(df),
    "paired": paired,
    "daily_bootstrap": daily_bootstrap,
    "artifacts": {
        "episodes_csv": str(csv_path),
        "report_md": str(out_dir / "dyfo_drl_walkforward_report.md"),
    },
}

json_path = out_dir / "dyfo_drl_walkforward_summary.json"
json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
_write_report(out_dir, payload)
print(f"\nEpisodes CSV -> {csv_path}")
print(f"Summary JSON -> {json_path}")
print(f"Report MD    -> {out_dir / 'dyfo_drl_walkforward_report.md'}")
return payload


def main() -> None:
parser = argparse.ArgumentParser(
    description="Walk-forward/bootstrap evidence for DyFO portfolio DRL.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--start", default=DATA_START)
parser.add_argument("--end", default=DATA_END)
parser.add_argument("--train_days", type=int, default=1000)
parser.add_argument("--val_days", type=int, default=250)
parser.add_argument("--test_days", type=int, default=125)
parser.add_argument("--step_days", type=int, default=125)
parser.add_argument("--max_windows", type=int, default=None)
parser.add_argument("--episode_len", type=int, default=EPISODE_LEN)
parser.add_argument("--train_episode_step", type=int, default=None)
parser.add_argument("--test_episode_step", type=int, default=None)
parser.add_argument("--drl_episodes", type=int, default=30)
parser.add_argument("--seeds", default="42,123,456")
parser.add_argument("--epochs", type=int, default=8)
parser.add_argument("--lr", type=float, default=TGN_LR)
parser.add_argument("--lr_drl", type=float, default=3e-4)
parser.add_argument("--n_heads", type=int, default=4)
parser.add_argument("--finetune_encoder", action="store_true")
parser.add_argument("--warmup_days", type=int, default=20)
parser.add_argument("--alpha_ewma", type=float, default=ALPHA_EWMA)
parser.add_argument("--n_bootstrap", type=int, default=1000)
parser.add_argument("--bootstrap_block_len", type=int, default=20)
parser.add_argument("--bootstrap_seed", type=int, default=2026)
parser.add_argument("--checkpoint_mode", choices=["causal", "reuse"], default="causal")
parser.add_argument("--checkpoint", default=str(RESULTS_DIR / "dyfo_portfolio_ckpt.pt"))
parser.add_argument("--force_retrain", action="store_true")
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--out_dir", default=str(RESULTS_DIR / "dyfo_drl_walkforward"))
args = parser.parse_args()
run(args)


if __name__ == "__main__":
main()
