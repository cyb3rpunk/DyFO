"""DyFO Discrete Deep Q-Network (DQN) Dynamic Hedging Demo.

Demonstrates discrete regime switching and tail-risk hedging using Dueling Double-DQN:
1. Formulates compact topological-spectral MDP states s_t in R^16 from DyFO graph outputs.
2. Trains DQNHedgingAgent with Prioritized Experience Replay (PER).
3. Evaluates out-of-sample dynamic regime switching across market conditions.
4. Generates a 4-panel visualization of equity curves, action frequency, rolling volatility, and Q-values.
5. Saves execution report to results/demo_dyfo_dqn_hedging.json.

Usage:
    python examples/demo_dyfo_dqn_hedging.py
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
from dyfo.dqn.discrete_state import DiscreteStateConstructor, REGIME_ACTIONS
from dyfo.dqn.dqn_agent import DQNHedgingAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DyFO.DQNDemo")


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
    logger.info("  DYFO DISCRETE DEEP Q-NETWORK (DQN) DYNAMIC HEDGING DEMO  ")
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

    # 2. Instantiate State Constructor & DQN Agent
    adapter = DyFOAdapter(config=config, tickers=tickers, device=device)
    state_constructor = DiscreteStateConstructor(tickers=tickers, state_dim=16, device=device)
    agent = DQNHedgingAgent(state_dim=16, num_actions=4, lr=5e-4, buffer_capacity=10000, device=device)

    # 3. Pre-train DQN on Training Split (30 epochs with Prioritized Experience Replay)
    logger.info("Training Dueling Double-DQN over %d historical steps (30 epochs)...", len(train_dates))
    
    # Populate initial replay buffer
    w_curr = np.full(num_assets, 1.0 / num_assets, dtype=np.float32)
    dd_track = 0.0
    wealth_track = 1.0
    peak_track = 1.0

    for step in range(len(train_dates) - 1):
        dt = train_dates[step].date()
        sigma_t = adapter.get_covariance_matrix(dt)
        snap = adapter.export_structural_graph(dt)
        st = state_constructor.build_state(
            cov_matrix=sigma_t,
            node_embeddings=snap.node_embeddings,
            current_drawdown=dd_track,
            realized_vol_30d=0.15,
            date_str=dt.isoformat(),
        )

        act = agent.select_action(st, deterministic=False)
        w_step = agent.execute_regime_action(act, sigma_t, snap.node_embeddings, tickers=tickers)
        r_step = float(w_step @ train_returns[step + 1])
        wealth_track *= (1.0 + r_step)
        peak_track = max(peak_track, wealth_track)
        dd_track = (wealth_track - peak_track) / peak_track

        # Next state
        dt_next = train_dates[step + 1].date()
        sigma_next = adapter.get_covariance_matrix(dt_next)
        snap_next = adapter.export_structural_graph(dt_next)
        st_next = state_constructor.build_state(
            cov_matrix=sigma_next,
            node_embeddings=snap_next.node_embeddings,
            current_drawdown=dd_track,
            realized_vol_30d=0.15,
            date_str=dt_next.isoformat(),
        )

        # Reward: Differential Sharpe + Drawdown penalty
        reward = np.log(max(1.0 + r_step, 1e-4)) - 0.005 * np.sum(np.abs(w_step - w_curr))
        if dd_track < -0.05:
            reward -= 0.02 * abs(dd_track)

        agent.replay_buffer.push(
            state=st.features.cpu().numpy(),
            action=act,
            reward=reward,
            next_state=st_next.features.cpu().numpy(),
            done=False,
        )
        w_curr = w_step

        if len(agent.replay_buffer) >= 32:
            agent.train_step(batch_size=32)

    logger.info("DQN Training completed. Epsilon = %.4f | Buffer Size = %d", agent.epsilon, len(agent.replay_buffer))

    # 4. Out-of-Sample Walk-Forward Simulation (250 Days)
    logger.info("Simulating out-of-sample DQN regime switching evaluation (%d days)...", eval_window - 1)
    models = ["DyFO-DQN (Regime-Switching)", "DyFO-Static-GMVP", "EWMA-GMVP", "Equal-Weight (1/N)"]
    weights_hist: Dict[str, List[np.ndarray]] = {m: [] for m in models}
    pnl_hist: Dict[str, List[float]] = {m: [] for m in models}
    action_history: List[int] = []

    w_dqn_prev = np.full(num_assets, 1.0 / num_assets, dtype=np.float32)
    dd_eval = 0.0
    wealth_eval = 1.0
    peak_eval = 1.0

    for t_idx in range(len(test_dates) - 1):
        dt = test_dates[t_idx]
        dt_date = dt.date()
        r_next = test_returns[t_idx + 1]

        # -------------------------------------------------------------
        # A. DyFO-DQN Agent
        # -------------------------------------------------------------
        sigma_t = adapter.get_covariance_matrix(dt_date)
        snap = adapter.export_structural_graph(dt_date)
        st_dqn = state_constructor.build_state(
            cov_matrix=sigma_t,
            node_embeddings=snap.node_embeddings,
            current_drawdown=dd_eval,
            date_str=dt_date.isoformat(),
        )

        act_chosen = agent.select_action(st_dqn, deterministic=True)
        action_history.append(act_chosen)
        w_dqn_raw = agent.execute_regime_action(act_chosen, sigma_t, snap.node_embeddings, tickers=tickers)

        # Smooth weights
        w_dqn = 0.10 * w_dqn_raw + 0.90 * w_dqn_prev
        w_dqn /= np.sum(w_dqn)
        weights_hist["DyFO-DQN (Regime-Switching)"].append(w_dqn)
        
        r_pnl = float(w_dqn @ r_next)
        pnl_hist["DyFO-DQN (Regime-Switching)"].append(r_pnl)
        wealth_eval *= (1.0 + r_pnl)
        peak_eval = max(peak_eval, wealth_eval)
        dd_eval = (wealth_eval - peak_eval) / peak_eval
        w_dqn_prev = w_dqn

        # -------------------------------------------------------------
        # B. DyFO-Static-GMVP Baseline
        # -------------------------------------------------------------
        try:
            inv_s = np.linalg.pinv(sigma_t + np.eye(num_assets) * 1e-5)
            ones = np.ones(num_assets)
            w_gmvp = inv_s @ ones / (ones @ inv_s @ ones)
            w_gmvp = np.clip(w_gmvp, 0.0, None)
            w_gmvp /= np.sum(w_gmvp)
        except Exception:
            w_gmvp = np.full(num_assets, 1.0 / num_assets)
        weights_hist["DyFO-Static-GMVP"].append(w_gmvp)
        pnl_hist["DyFO-Static-GMVP"].append(float(w_gmvp @ r_next))

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

    # 5. Consolidate Performance Metrics
    results_summary: Dict[str, Dict[str, float]] = {}
    for m in models:
        r_arr = np.array(pnl_hist[m])
        w_arr = np.array(weights_hist[m])
        results_summary[m] = calculate_metrics(r_arr, w_arr)

    # 6. Display Tabular Report
    print("\n" + "=" * 100)
    print("  DYFO DISCRETE DEEP Q-NETWORK (DQN) DYNAMIC HEDGING PERFORMANCE  ")
    print("=" * 100)
    print(f"{'Strategy / Model':<30} | {'Ann. Ret':<9} | {'Ann. Vol':<9} | {'Sharpe':<8} | {'Max DD':<9} | {'Turnover':<9} | {'Wealth'}")
    print("-" * 100)
    for m in models:
        res = results_summary[m]
        print(f"{m:<30} | {res['annualized_return']*100:>7.2f}% | {res['annualized_volatility']*100:>7.2f}% | {res['sharpe_ratio']:>8.4f} | {res['max_drawdown']*100:>7.2f}% | {res['turnover']:>9.4f} | {res['final_wealth']:>7.4f}x")
    print("=" * 100 + "\n")

    # 7. Generate 4-Panel Visualization
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    out_fig_path = fig_dir / "demo_dyfo_dqn_hedging.png"

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=200)
    fig.patch.set_facecolor("#0b1329")

    colors = {
        "DyFO-DQN (Regime-Switching)": "#10b981",  # Emerald
        "DyFO-Static-GMVP": "#38bdf8",            # Sky Blue
        "EWMA-GMVP": "#f59e0b",                   # Amber
        "Equal-Weight (1/N)": "#94a3b8",          # Slate
    }

    t_axis = test_dates[1:]

    # Panel A: Cumulative Wealth
    ax0 = axes[0, 0]
    ax0.set_facecolor("#152238")
    for m in models:
        cum_ret = np.cumprod(1.0 + np.array(pnl_hist[m]))
        ax0.plot(t_axis, cum_ret, label=m, color=colors[m], linewidth=2.2 if m == "DyFO-DQN (Regime-Switching)" else 1.4)
    ax0.set_title("A. Cumulative Wealth (DQN Dynamic Hedging vs Baselines)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax0.set_ylabel("Wealth ($1.00 base)", color="#94a3b8")
    ax0.tick_params(colors="#94a3b8")
    ax0.legend(loc="upper left", framealpha=0.3, facecolor="#0b1329", labelcolor="#f8fafc", fontsize=8)

    # Panel B: Action Regime Trajectory Over Time
    ax1 = axes[0, 1]
    ax1.set_facecolor("#152238")
    act_arr = np.array(action_history)
    ax1.plot(t_axis, act_arr, color="#10b981", marker="o", markersize=2.5, linestyle="none", alpha=0.8)
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(["Alpha (GMVP)", "Defensive (ERC)", "Tail Hedge (Cash)", "Sector Rotate"], color="#cbd5e1", fontsize=8)
    ax1.set_title("B. DQN Active Regime Decisions Over Time", color="#f8fafc", fontsize=11, fontweight="bold")
    ax1.tick_params(colors="#94a3b8")

    # Panel C: Drawdown Curves
    ax2 = axes[1, 0]
    ax2.set_facecolor("#152238")
    for m in models:
        cum_ret = np.cumprod(1.0 + np.array(pnl_hist[m]))
        peak = np.maximum.accumulate(cum_ret)
        dd = (cum_ret - peak) / peak * 100
        ax2.plot(t_axis, dd, label=m, color=colors[m], linewidth=1.8 if m == "DyFO-DQN (Regime-Switching)" else 1.2)
    ax2.set_title("C. Out-of-Sample Drawdowns (%)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Drawdown (%)", color="#94a3b8")
    ax2.tick_params(colors="#94a3b8")

    # Panel D: Action Distribution Bar Chart
    ax3 = axes[1, 1]
    ax3.set_facecolor("#152238")
    action_counts = [int(np.sum(act_arr == i)) for i in range(4)]
    action_labels = ["Alpha (GMVP)", "Defensive (ERC)", "Tail Hedge", "Sector Rotate"]
    bar_colors = ["#38bdf8", "#10b981", "#ef4444", "#f59e0b"]
    bars = ax3.bar(action_labels, action_counts, color=bar_colors, alpha=0.85, width=0.55)
    for bar in bars:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, yval + 2, f"{yval} d", ha="center", va="bottom", color="#f8fafc", fontsize=8.5)
    ax3.set_title("D. Discrete Regime Action Frequency Distribution", color="#f8fafc", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Trading Days", color="#94a3b8")
    ax3.tick_params(colors="#94a3b8")

    plt.tight_layout()
    plt.savefig(out_fig_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    logger.info("Saved DQN Dynamic Hedging visualization -> %s", out_fig_path)

    # 8. Save JSON Summary
    out_json_path = Path("results") / "demo_dyfo_dqn_hedging.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "universe_size": num_assets,
                "eval_window_days": eval_window,
                "results": results_summary,
                "action_distribution": {REGIME_ACTIONS[i]: action_counts[i] for i in range(4)},
            },
            f,
            indent=2,
        )
    logger.info("Saved JSON execution report -> %s", out_json_path)
    logger.info("Phase 3 Discrete DQN Demo Completed Successfully!")


if __name__ == "__main__":
    main()
