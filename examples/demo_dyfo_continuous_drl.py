"""DyFO Continuous Deep Reinforcement Learning (Actor-Critic / PPO) Demo.

Demonstrates continuous portfolio optimization using DyFO relational state embeddings:
1. Constructs relational state tokens combining DyFO node embeddings Z_t, w_{t-1}, and macro regime.
2. Trains RelationalActorCriticPolicy using Risk-Regularized PPO (penalizing turnover, variance, DD).
3. Compares out-of-sample against Raw-DRL (no graph), EWMA-GMVP, and Equal-Weight (1/N).
4. Exports a 4-panel visualization and JSON execution report.

Usage:
    python examples/demo_dyfo_continuous_drl.py
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dyfo.adapters.dyfo_adapter import DyFOAdapter
from dyfo.config import DyFOConfig
from dyfo.core.link_prediction import project_to_spd_correlation
from dyfo.core.ticker_registry import TICKERS_30, TICKER_GICS_MAPPING, get_sector_distribution
from dyfo.drl.continuous_state import ContinuousStateConstructor
from dyfo.drl.ppo_trainer import EpisodeTrajectory, PPOConfig, PPOTrainer
from dyfo.drl.relational_actor_critic import RelationalActorCriticPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DyFO.DRLDemo")


def calculate_metrics(returns: np.ndarray, weights_history: np.ndarray) -> Dict[str, float]:
    """Compute financial KPIs for return stream."""
    returns = np.nan_to_num(returns, nan=0.0)
    ann_factor = 252.0
    mean_ret = float(np.mean(returns)) * ann_factor
    vol = float(np.std(returns)) * np.sqrt(ann_factor)
    sharpe = mean_ret / (vol + 1e-8)
    wealth = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(wealth)
    drawdowns = (wealth - peak) / peak
    mdd = float(np.min(drawdowns))

    if len(weights_history) > 1:
        diffs = np.abs(weights_history[1:] - weights_history[:-1])
        turnover = float(np.mean(np.sum(diffs, axis=1)))
    else:
        turnover = 0.0

    return {
        "annualized_return": mean_ret,
        "annualized_volatility": vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
        "turnover": turnover,
        "final_wealth": float(wealth[-1]),
    }


def main():
    logger.info("=" * 75)
    logger.info("  DYFO CONTINUOUS DEEP REINFORCEMENT LEARNING (PPO) DEMO  ")
    logger.info("=" * 75)

    config = DyFOConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using compute device: %s", device)

    # 1. Load Causal Market Data
    cache_candidates = list(Path("results").glob("prepared_data_cache_*.pkl"))
    data = None
    if cache_candidates:
        import pickle
        chosen_cache = cache_candidates[0]
        for c in cache_candidates:
            if "c22b639f92" in c.name:
                chosen_cache = c
                break
        with open(chosen_cache, "rb") as f:
            data = pickle.load(f)

    if data is None:
        from scripts.train_link_prediction import prepare_data
        data = prepare_data(config)

    prices: pd.DataFrame = data["prices"]
    tickers = list(prices.columns)
    num_assets = len(tickers)
    logger.info("Assets in Universe: N = %d (%s...)", num_assets, ", ".join(tickers[:5]))

    dates = prices.index
    eval_window = 250
    train_window = 500
    
    train_dates = dates[-(eval_window + train_window):-eval_window]
    test_dates = dates[-eval_window:]

    returns = prices.pct_change().dropna()
    train_returns = returns.loc[train_dates].values
    test_returns = returns.loc[test_dates].values

    # 2. Instantiate State Constructor & Adapter
    adapter = DyFOAdapter(config=config, tickers=tickers, device=device)
    state_constructor = ContinuousStateConstructor(tickers=tickers, embedding_dim=100, macro_dim=4, device=device)

    # 3. Build & Train DyFO-DRL Policy
    logger.info("Initializing Relational Actor-Critic Policy (Cross-Attention Transformer)...")
    policy_dyfo = RelationalActorCriticPolicy(feature_dim=105, hidden_dim=48, num_heads=4, num_layers=2).to(device)
    trainer_dyfo = PPOTrainer(policy=policy_dyfo, config=PPOConfig(lr=5e-4, turnover_penalty=0.005), device=device)

    # Quick pre-training episodic loop on train_dates (15 epochs)
    logger.info("Training DyFO-DRL policy over %d historical training steps (15 PPO epochs)...", len(train_dates))
    for ep in range(15):
        traj = EpisodeTrajectory()
        w_curr = np.full(num_assets, 1.0 / num_assets, dtype=np.float32)
        
        # Sample sub-window of 60 days per episode
        start_idx = np.random.randint(0, max(1, len(train_dates) - 60))
        for step in range(start_idx, min(len(train_dates) - 1, start_idx + 60)):
            dt = train_dates[step].date()
            sigma_t = adapter.get_covariance_matrix(dt)
            # Node embeddings from adapter structural snapshot
            snap = adapter.export_structural_graph(dt)
            embs = snap.node_embeddings
            st = state_constructor.build_state(embs, w_curr, date_str=dt.isoformat())

            w_next, lp, val = policy_dyfo.act(st, deterministic=False)
            r_realized = float(w_next @ train_returns[step + 1])
            reward = trainer_dyfo.compute_step_reward(r_realized, w_next, w_curr, cov_matrix=sigma_t)

            traj.states.append(st)
            traj.actions.append(w_next)
            traj.log_probs.append(lp)
            traj.rewards.append(reward)
            traj.values.append(val)
            traj.dones.append(False)
            w_curr = w_next

        metrics = trainer_dyfo.train_epoch(traj, ppo_epochs=3)
        if (ep + 1) % 5 == 0:
            logger.info("Episode [%2d/15] - PPO Loss: %.4f | Policy Loss: %.4f | Mean Reward: %.4f", ep + 1, metrics["loss"], metrics["policy_loss"], metrics["mean_reward"])

    # 4. Build & Train Raw-DRL Policy (Ablation Baseline: No Graph)
    policy_raw = RelationalActorCriticPolicy(feature_dim=105, hidden_dim=48, num_heads=4, num_layers=2).to(device)
    trainer_raw = PPOTrainer(policy=policy_raw, config=PPOConfig(lr=5e-4, turnover_penalty=0.005), device=device)

    logger.info("Training Raw-DRL policy (without graph embeddings)...")
    for ep in range(15):
        traj_raw = EpisodeTrajectory()
        w_curr = np.full(num_assets, 1.0 / num_assets, dtype=np.float32)
        start_idx = np.random.randint(0, max(1, len(train_dates) - 60))
        for step in range(start_idx, min(len(train_dates) - 1, start_idx + 60)):
            p_slice = prices.loc[train_dates[:step + 1]].values
            st_raw = state_constructor.build_raw_state(p_slice, w_curr, date_str=train_dates[step].date().isoformat())
            w_next, lp, val = policy_raw.act(st_raw, deterministic=False)
            r_realized = float(w_next @ train_returns[step + 1])
            reward = trainer_raw.compute_step_reward(r_realized, w_next, w_curr)

            traj_raw.states.append(st_raw)
            traj_raw.actions.append(w_next)
            traj_raw.log_probs.append(lp)
            traj_raw.rewards.append(reward)
            traj_raw.values.append(val)
            traj_raw.dones.append(False)
            w_curr = w_next
        trainer_raw.train_epoch(traj_raw, ppo_epochs=3)

    # 5. Out-of-Sample Walk-Forward Simulation (250 Days)
    logger.info("Simulating out-of-sample DRL walk-forward evaluation (%d days)...", eval_window - 1)
    models = ["DyFO-DRL (Relational)", "Raw-DRL (No Graph)", "EWMA-GMVP", "Equal-Weight (1/N)"]
    weights_hist: Dict[str, List[np.ndarray]] = {m: [] for m in models}
    pnl_hist: Dict[str, List[float]] = {m: [] for m in models}

    w_dyfo_prev = np.full(num_assets, 1.0 / num_assets, dtype=np.float32)
    w_raw_prev = np.full(num_assets, 1.0 / num_assets, dtype=np.float32)

    for t_idx in range(len(test_dates) - 1):
        dt = test_dates[t_idx]
        dt_date = dt.date()
        r_next = test_returns[t_idx + 1]

        # -------------------------------------------------------------
        # A. DyFO-DRL (Graph-Augmented State)
        # -------------------------------------------------------------
        snap = adapter.export_structural_graph(dt_date)
        st_dyfo = state_constructor.build_state(snap.node_embeddings, w_dyfo_prev, date_str=dt_date.isoformat())
        w_dyfo_raw, _, _ = policy_dyfo.act(st_dyfo, deterministic=True)
        # Apply smoothing
        w_dyfo = 0.08 * w_dyfo_raw + 0.92 * w_dyfo_prev
        w_dyfo /= np.sum(w_dyfo)
        weights_hist["DyFO-DRL (Relational)"].append(w_dyfo)
        pnl_hist["DyFO-DRL (Relational)"].append(float(w_dyfo @ r_next))
        w_dyfo_prev = w_dyfo

        # -------------------------------------------------------------
        # B. Raw-DRL (Flat Price Features, No Graph)
        # -------------------------------------------------------------
        curr_p_slice = prices.loc[:dt].values
        st_raw = state_constructor.build_raw_state(curr_p_slice, w_raw_prev, date_str=dt_date.isoformat())
        w_raw_act, _, _ = policy_raw.act(st_raw, deterministic=True)
        w_raw = 0.08 * w_raw_act + 0.92 * w_raw_prev
        w_raw /= np.sum(w_raw)
        weights_hist["Raw-DRL (No Graph)"].append(w_raw)
        pnl_hist["Raw-DRL (No Graph)"].append(float(w_raw @ r_next))
        w_raw_prev = w_raw

        # -------------------------------------------------------------
        # C. EWMA-GMVP Baseline
        # -------------------------------------------------------------
        lookback_idx = max(0, t_idx + len(dates) - eval_window - 63)
        curr_idx = t_idx + len(dates) - eval_window
        hist_ret = returns.iloc[lookback_idx:curr_idx].values
        decay = 0.94
        w_decay = (1 - decay) * (decay ** np.arange(hist_ret.shape[0])[::-1])
        w_decay /= np.sum(w_decay)
        cov_ewma = (hist_ret - hist_ret.mean(axis=0)).T @ np.diag(w_decay) @ (hist_ret - hist_ret.mean(axis=0)) * 252
        sigma_ewma = project_to_spd_correlation(cov_ewma)
        try:
            inv_cov = np.linalg.pinv(sigma_ewma + np.eye(num_assets) * 1e-5)
            ones = np.ones(num_assets)
            w_ewma = inv_cov @ ones / (ones @ inv_cov @ ones)
            w_ewma = np.clip(w_ewma, 0.0, None)
            w_ewma /= np.sum(w_ewma)
        except Exception:
            w_ewma = np.full(num_assets, 1.0 / num_assets)
        weights_hist["EWMA-GMVP"].append(w_ewma)
        pnl_hist["EWMA-GMVP"].append(float(w_ewma @ r_next))

        # -------------------------------------------------------------
        # D. Equal-Weight (1/N)
        # -------------------------------------------------------------
        w_eq = np.full(num_assets, 1.0 / num_assets)
        weights_hist["Equal-Weight (1/N)"].append(w_eq)
        pnl_hist["Equal-Weight (1/N)"].append(float(w_eq @ r_next))

    # 6. Consolidate Performance Metrics
    results_summary: Dict[str, Dict[str, float]] = {}
    for m in models:
        r_arr = np.array(pnl_hist[m])
        w_arr = np.array(weights_hist[m])
        results_summary[m] = calculate_metrics(r_arr, w_arr)

    # 7. Display Tabular Report
    print("\n" + "=" * 100)
    print("  DYFO CONTINUOUS DEEP REINFORCEMENT LEARNING (PPO) PERFORMANCE (1-YEAR TEST)  ")
    print("=" * 100)
    print(f"{'Strategy / Model':<28} | {'Ann. Ret':<9} | {'Ann. Vol':<9} | {'Sharpe':<8} | {'Max DD':<9} | {'Turnover':<9} | {'Wealth'}")
    print("-" * 100)
    for m in models:
        res = results_summary[m]
        print(f"{m:<28} | {res['annualized_return']*100:>7.2f}% | {res['annualized_volatility']*100:>7.2f}% | {res['sharpe_ratio']:>8.4f} | {res['max_drawdown']*100:>7.2f}% | {res['turnover']:>9.4f} | {res['final_wealth']:>7.4f}x")
    print("=" * 100 + "\n")

    # 8. Generate 4-Panel Visualization
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    out_fig_path = fig_dir / "demo_dyfo_continuous_drl.png"

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=200)
    fig.patch.set_facecolor("#0b1329")

    colors = {
        "DyFO-DRL (Relational)": "#10b981",  # Emerald
        "Raw-DRL (No Graph)": "#ef4444",      # Rose/Red
        "EWMA-GMVP": "#f59e0b",              # Amber
        "Equal-Weight (1/N)": "#94a3b8",      # Slate
    }

    t_axis = test_dates[1:]

    # Panel A: Cumulative Wealth
    ax0 = axes[0, 0]
    ax0.set_facecolor("#152238")
    for m in models:
        cum_ret = np.cumprod(1.0 + np.array(pnl_hist[m]))
        ax0.plot(t_axis, cum_ret, label=m, color=colors[m], linewidth=2.2 if m == "DyFO-DRL (Relational)" else 1.4)
    ax0.set_title("A. Cumulative Wealth (DyFO-DRL vs Raw-DRL vs Baselines)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax0.set_ylabel("Wealth ($1.00 base)", color="#94a3b8")
    ax0.tick_params(colors="#94a3b8")
    ax0.legend(loc="upper left", framealpha=0.3, facecolor="#0b1329", labelcolor="#f8fafc", fontsize=8)

    # Panel B: Rolling 30-Day Realized Volatility
    ax1 = axes[0, 1]
    ax1.set_facecolor("#152238")
    for m in models:
        roll_vol = pd.Series(pnl_hist[m], index=t_axis).rolling(30).std() * np.sqrt(252) * 100
        ax1.plot(t_axis, roll_vol, label=m, color=colors[m], linewidth=2.0 if m == "DyFO-DRL (Relational)" else 1.2)
    ax1.set_title("B. Rolling 30-Day Realized Volatility (%)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Annualized Volatility (%)", color="#94a3b8")
    ax1.tick_params(colors="#94a3b8")

    # Panel C: Drawdown Trajectories
    ax2 = axes[1, 0]
    ax2.set_facecolor("#152238")
    for m in models:
        cum_ret = np.cumprod(1.0 + np.array(pnl_hist[m]))
        peak = np.maximum.accumulate(cum_ret)
        dd = (cum_ret - peak) / peak * 100
        ax2.plot(t_axis, dd, label=m, color=colors[m], linewidth=1.8 if m == "DyFO-DRL (Relational)" else 1.2)
    ax2.set_title("C. Out-of-Sample Drawdowns (%) - Symmetry vs Collapse", color="#f8fafc", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Drawdown (%)", color="#94a3b8")
    ax2.tick_params(colors="#94a3b8")

    # Panel D: DyFO-DRL Sector Allocations
    ax3 = axes[1, 1]
    ax3.set_facecolor("#152238")
    w_drl_arr = np.array(weights_hist["DyFO-DRL (Relational)"])
    sector_dist = get_sector_distribution(tickers)
    palette = ["#10b981", "#38bdf8", "#f59e0b", "#a855f7", "#ec4899", "#6366f1"]
    for idx_color, (sector, count) in enumerate(list(sector_dist.items())[:5]):
        sec_idx = [i for i, t in enumerate(tickers) if TICKER_GICS_MAPPING.get(t) == sector]
        sec_weight = np.sum(w_drl_arr[:, sec_idx], axis=1) * 100
        ax3.plot(t_axis, sec_weight, label=f"{sector} ({count} assets)", color=palette[idx_color % len(palette)], linewidth=1.6)
    ax3.set_title("D. DyFO-DRL Learned Sector Allocations (%)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Sector Weight (%)", color="#94a3b8")
    ax3.tick_params(colors="#94a3b8")
    ax3.legend(loc="upper right", framealpha=0.3, facecolor="#0b1329", labelcolor="#f8fafc", fontsize=7.5)

    plt.tight_layout()
    plt.savefig(out_fig_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    logger.info("Saved DyFO-DRL visualization -> %s", out_fig_path)

    # 9. Save JSON Execution Summary
    out_json_path = Path("results") / "demo_dyfo_continuous_drl.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "universe_size": num_assets,
                "eval_window_days": eval_window,
                "results": results_summary,
            },
            f,
            indent=2,
        )
    logger.info("Saved JSON execution report -> %s", out_json_path)
    logger.info("Phase 2 Continuous DRL Demo Completed Successfully!")


if __name__ == "__main__":
    main()
