"""DyFO Neuro-Symbolic AI & GraphRAG LLM Reasoning Demo.

Demonstrates end-to-end integration of DyFO's dynamic temporal graph co-movement
predictions with Neuro-Symbolic AI and Large Language Model (LLM) reasoning:

1. Extracts causal subgraphs and typed triples on daily market walk-forward steps.
2. Invokes LLM reasoning engine to generate causal Chain-of-Thought risk explanations.
3. Compiles high-level symbolic risk decisions into linear inequality constraints.
4. Executes symbolically-constrained GMVP optimization with Higham SPD projections.
5. Evaluates out-of-sample risk reduction and exports visual charts and JSON reports.

Usage:
    python examples/demo_dyfo_llm_neurosymbolic.py
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

from dyfo.adapters.dyfo_adapter import DyFOAdapter
from dyfo.config import DyFOConfig
from dyfo.core.link_prediction import project_to_spd_correlation
from dyfo.core.ticker_registry import TICKERS_30, TICKER_GICS_MAPPING, get_sector_distribution
from dyfo.neurosymbolic.constrained_solver import ConstrainedPortfolioSolver
from dyfo.neurosymbolic.graphrag_prompt_engine import LLMReasoner, RiskExplanation
from dyfo.neurosymbolic.subgraph_extractor import CausalSubgraphExtractor
from dyfo.neurosymbolic.symbolic_parser import SymbolicConstraintParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DyFO.NeuroSymbolicDemo")


def calculate_metrics(
    returns: np.ndarray,
    weights_history: np.ndarray,
    tx_cost_bps: float = 10.0,
) -> Dict[str, float]:
    """Compute financial KPIs for return stream including transaction costs."""
    returns = np.nan_to_num(returns, nan=0.0)
    ann_factor = 252.0
    c_tx = tx_cost_bps * 1e-4  # 10 bps = 0.0010

    # Daily turnover & transaction cost drag
    t_steps = len(returns)
    if len(weights_history) > 1:
        diffs = np.abs(weights_history[1:] - weights_history[:-1])
        daily_turnover = np.concatenate([[0.0], np.sum(diffs, axis=1)])[:t_steps]
        mean_turnover = float(np.mean(daily_turnover[1:]))
    else:
        daily_turnover = np.zeros(t_steps)
        mean_turnover = 0.0

    daily_tx_cost = c_tx * daily_turnover
    net_returns = returns - daily_tx_cost

    # Gross & Net Metrics
    gross_ret = float(np.mean(returns)) * ann_factor
    gross_vol = float(np.std(returns)) * np.sqrt(ann_factor)
    gross_sharpe = gross_ret / (gross_vol + 1e-8)
    gross_wealth = np.cumprod(1.0 + returns)
    gross_peak = np.maximum.accumulate(gross_wealth)
    gross_mdd = float(np.min((gross_wealth - gross_peak) / gross_peak))

    net_ret = float(np.mean(net_returns)) * ann_factor
    net_vol = float(np.std(net_returns)) * np.sqrt(ann_factor)
    net_sharpe = net_ret / (net_vol + 1e-8)
    net_wealth = np.cumprod(1.0 + net_returns)
    net_peak = np.maximum.accumulate(net_wealth)
    net_mdd = float(np.min((net_wealth - net_peak) / net_peak))
    ann_cost_drag_bps = float(np.mean(daily_tx_cost) * ann_factor * 10000.0)

    return {
        "annualized_gross_return": gross_ret,
        "annualized_net_return": net_ret,
        "annualized_volatility": net_vol,
        "gross_sharpe_ratio": gross_sharpe,
        "net_sharpe_ratio": net_sharpe,
        "max_drawdown": net_mdd,
        "turnover": mean_turnover,
        "annualized_cost_drag_bps": ann_cost_drag_bps,
        "final_gross_wealth": float(gross_wealth[-1]),
        "final_net_wealth": float(net_wealth[-1]),
        # backward compatibility
        "annualized_return": net_ret,
        "sharpe_ratio": net_sharpe,
        "final_wealth": float(net_wealth[-1]),
    }


def main():
    logger.info("=" * 75)
    logger.info("  DYFO NEURO-SYMBOLIC AI & GRAPHRAG LLM REASONING DEMO  ")
    logger.info("=" * 75)

    config = DyFOConfig()

    # 1. Load Causal Market Data
    cache_candidates = list(Path("results").glob("prepared_data_cache_*.pkl"))
    data = None
    if cache_candidates:
        import pickle
        # Prefer cache with 30 tickers if present
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
    test_dates = dates[-eval_window:]
    returns = prices.pct_change().dropna()
    test_returns = returns.loc[test_dates].values

    # 2. Instantiate DyFO Adapter & Neuro-Symbolic Pipeline
    adapter = DyFOAdapter(config=config, tickers=tickers)
    extractor = CausalSubgraphExtractor(tickers=tickers, sector_mapping=TICKER_GICS_MAPPING)
    reasoner = LLMReasoner(backend="mock")
    parser = SymbolicConstraintParser(tickers=tickers, sector_mapping=TICKER_GICS_MAPPING)
    solver = ConstrainedPortfolioSolver()

    # 3. Simulate Multi-Strategy Out-of-Sample Walk-Forward
    strategies = ["DyFO-NeuroSymbolic-LLM", "DyFO-Unconstrained", "EWMA-GMVP", "Equal-Weight (1/N)"]
    weights_hist: Dict[str, List[np.ndarray]] = {s: [] for s in strategies}
    pnl_hist: Dict[str, List[float]] = {s: [] for s in strategies}
    explanations_log: List[Dict] = []

    logger.info("Simulating daily Neuro-Symbolic reasoning across %d trading days...", eval_window - 1)

    for t_idx in range(len(test_dates) - 1):
        dt = test_dates[t_idx]
        dt_date = dt.date()
        date_str = dt_date.isoformat()
        r_next = test_returns[t_idx + 1]

        # A. DyFO Covariance & Delta Correlation
        sigma_dyfo = adapter.get_covariance_matrix(dt_date)
        
        # Synthetic / predicted delta rho approximation from covariance evolution
        diag_inv = 1.0 / np.sqrt(np.diag(sigma_dyfo) + 1e-6)
        corr_today = np.outer(diag_inv, diag_inv) * sigma_dyfo
        delta_rho_sim = (corr_today - np.eye(num_assets)) * 0.10

        # -------------------------------------------------------------
        # 1. Neuro-Symbolic LLM Pipeline (Extract -> Reason -> Parse -> Solve)
        # -------------------------------------------------------------
        subgraph = extractor.extract_subgraph(
            date_str=date_str,
            predicted_delta_rho=delta_rho_sim,
            correlation_matrix=corr_today,
            top_k_shocks=12,
        )

        explanation = reasoner.reason(subgraph)
        constraints = parser.parse(explanation)
        w_sym, meta_sym = solver.solve(sigma_dyfo, constraints)

        # Apply smoothing buffer
        if weights_hist["DyFO-NeuroSymbolic-LLM"]:
            w_sym_smooth = 0.10 * w_sym + 0.90 * weights_hist["DyFO-NeuroSymbolic-LLM"][-1]
            w_sym_smooth = w_sym_smooth / np.sum(w_sym_smooth) * (1.0 - constraints.cash_buffer)
        else:
            w_sym_smooth = w_sym

        weights_hist["DyFO-NeuroSymbolic-LLM"].append(w_sym_smooth)
        pnl_hist["DyFO-NeuroSymbolic-LLM"].append(float(w_sym_smooth @ r_next))

        if t_idx % 50 == 0:
            explanations_log.append({
                "date": date_str,
                "macro_regime": subgraph.macro_regime,
                "eigen_concentration": round(subgraph.eigen_concentration, 4),
                "rationale": explanation.macro_rationale,
                "sector_caps": explanation.recommended_sector_caps,
                "cash_buffer": explanation.min_cash_buffer,
            })
            logger.info("[%s] Regime: %s | Eigen Conc: %.1f%% | LLM Action: %s", date_str, subgraph.macro_regime, subgraph.eigen_concentration * 100, explanation.hedging_action)

        # -------------------------------------------------------------
        # 2. Unconstrained DyFO GMVP
        # -------------------------------------------------------------
        w_raw, _ = solver.solve(sigma_dyfo, constraints=None, target_cash_buffer=0.0)
        if weights_hist["DyFO-Unconstrained"]:
            w_raw = 0.10 * w_raw + 0.90 * weights_hist["DyFO-Unconstrained"][-1]
            w_raw /= np.sum(w_raw)
        weights_hist["DyFO-Unconstrained"].append(w_raw)
        pnl_hist["DyFO-Unconstrained"].append(float(w_raw @ r_next))

        # -------------------------------------------------------------
        # 3. EWMA Baseline
        # -------------------------------------------------------------
        lookback_idx = max(0, t_idx + len(dates) - eval_window - 63)
        curr_idx = t_idx + len(dates) - eval_window
        hist_ret = returns.iloc[lookback_idx:curr_idx].values
        decay = 0.94
        w_decay = (1 - decay) * (decay ** np.arange(hist_ret.shape[0])[::-1])
        w_decay /= np.sum(w_decay)
        cov_ewma = (hist_ret - hist_ret.mean(axis=0)).T @ np.diag(w_decay) @ (hist_ret - hist_ret.mean(axis=0)) * 252
        sigma_ewma = project_to_spd_correlation(cov_ewma)
        w_ewma, _ = solver.solve(sigma_ewma, constraints=None, target_cash_buffer=0.0)
        weights_hist["EWMA-GMVP"].append(w_ewma)
        pnl_hist["EWMA-GMVP"].append(float(w_ewma @ r_next))

        # -------------------------------------------------------------
        # 4. Equal Weight (1/N)
        # -------------------------------------------------------------
        w_eq = np.full(num_assets, 1.0 / num_assets)
        weights_hist["Equal-Weight (1/N)"].append(w_eq)
        pnl_hist["Equal-Weight (1/N)"].append(float(w_eq @ r_next))

    # 4. Consolidate Performance Metrics
    results_summary: Dict[str, Dict[str, float]] = {}
    for s in strategies:
        r_arr = np.array(pnl_hist[s])
        w_arr = np.array(weights_hist[s])
        results_summary[s] = calculate_metrics(r_arr, w_arr)

    # 5. Display Tabular Report
    print("\n" + "=" * 115)
    print("  DYFO NEURO-SYMBOLIC AI & GRAPHRAG LLM PORTFOLIO PERFORMANCE (10 BPS TX COST)  ")
    print("=" * 115)
    print(f"{'Strategy / Model':<28} | {'Gross Ret':<9} | {'Net Ret':<9} | {'Net Vol':<9} | {'Net Sharpe':<10} | {'Max DD':<9} | {'Turnover':<9} | {'Cost Drag'}")
    print("-" * 115)
    for s in strategies:
        res = results_summary[s]
        print(f"{s:<28} | {res['annualized_gross_return']*100:>7.2f}% | {res['annualized_net_return']*100:>7.2f}% | {res['annualized_volatility']*100:>7.2f}% | {res['net_sharpe_ratio']:>10.4f} | {res['max_drawdown']*100:>7.2f}% | {res['turnover']:>9.4f} | {res['annualized_cost_drag_bps']:>6.1f} bps")
    print("=" * 115 + "\n")

    # 6. Generate 4-Panel Visualization
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    out_fig_path = fig_dir / "demo_dyfo_llm_neurosymbolic.png"

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=200)
    fig.patch.set_facecolor("#0b1329")

    colors = {
        "DyFO-NeuroSymbolic-LLM": "#10b981",  # Emerald
        "DyFO-Unconstrained": "#38bdf8",      # Sky Blue
        "EWMA-GMVP": "#f59e0b",              # Amber
        "Equal-Weight (1/N)": "#94a3b8",      # Slate
    }

    t_axis = test_dates[1:]

    # Panel A: Cumulative Wealth
    ax0 = axes[0, 0]
    ax0.set_facecolor("#152238")
    for s in strategies:
        cum_ret = np.cumprod(1.0 + np.array(pnl_hist[s]))
        ax0.plot(t_axis, cum_ret, label=s, color=colors[s], linewidth=2.2 if s == "DyFO-NeuroSymbolic-LLM" else 1.4)
    ax0.set_title("A. Cumulative Wealth (DyFO Neuro-Symbolic LLM vs Baselines)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax0.set_ylabel("Wealth ($1.00 base)", color="#94a3b8")
    ax0.tick_params(colors="#94a3b8")
    ax0.legend(loc="upper left", framealpha=0.3, facecolor="#0b1329", labelcolor="#f8fafc", fontsize=8)

    # Panel B: Rolling 30-Day Volatility
    ax1 = axes[0, 1]
    ax1.set_facecolor("#152238")
    for s in strategies:
        roll_vol = pd.Series(pnl_hist[s], index=t_axis).rolling(30).std() * np.sqrt(252) * 100
        ax1.plot(t_axis, roll_vol, label=s, color=colors[s], linewidth=2.0 if s == "DyFO-NeuroSymbolic-LLM" else 1.2)
    ax1.set_title("B. Rolling 30-Day Realized Volatility (%)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Annualized Volatility (%)", color="#94a3b8")
    ax1.tick_params(colors="#94a3b8")

    # Panel C: Drawdown Curves
    ax2 = axes[1, 0]
    ax2.set_facecolor("#152238")
    for s in strategies:
        cum_ret = np.cumprod(1.0 + np.array(pnl_hist[s]))
        peak = np.maximum.accumulate(cum_ret)
        dd = (cum_ret - peak) / peak * 100
        ax2.plot(t_axis, dd, label=s, color=colors[s], linewidth=1.8 if s == "DyFO-NeuroSymbolic-LLM" else 1.2)
    ax2.set_title("C. Out-of-Sample Drawdowns (%) - Tail-Risk Protection", color="#f8fafc", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Drawdown (%)", color="#94a3b8")
    ax2.tick_params(colors="#94a3b8")

    # Panel D: Sector Weight Distribution under LLM Constraints
    ax3 = axes[1, 1]
    ax3.set_facecolor("#152238")
    w_sym_arr = np.array(weights_hist["DyFO-NeuroSymbolic-LLM"])
    sector_dist = get_sector_distribution(tickers)
    palette = ["#10b981", "#38bdf8", "#f59e0b", "#a855f7", "#ec4899", "#6366f1"]
    for idx_color, (sector, count) in enumerate(list(sector_dist.items())[:5]):
        sec_idx = [i for i, t in enumerate(tickers) if TICKER_GICS_MAPPING.get(t) == sector]
        sec_weight = np.sum(w_sym_arr[:, sec_idx], axis=1) * 100
        ax3.plot(t_axis, sec_weight, label=f"{sector} ({count} assets)", color=palette[idx_color % len(palette)], linewidth=1.6)
    ax3.set_title("D. LLM-Constrained Sector Allocations Over Time (%)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Aggregate Sector Weight (%)", color="#94a3b8")
    ax3.tick_params(colors="#94a3b8")
    ax3.legend(loc="upper right", framealpha=0.3, facecolor="#0b1329", labelcolor="#f8fafc", fontsize=7.5)

    plt.tight_layout()
    plt.savefig(out_fig_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    logger.info("Saved Neuro-Symbolic visualization -> %s", out_fig_path)

    # 7. Save JSON Execution Summary
    out_json_path = Path("results") / "demo_dyfo_llm_neurosymbolic.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.datetime.now().isoformat(),
                "universe_size": num_assets,
                "test_window_days": eval_window,
                "results": results_summary,
                "sample_explanations": explanations_log,
            },
            f,
            indent=2,
        )
    logger.info("Saved JSON execution report -> %s", out_json_path)
    logger.info("Phase 1 Neuro-Symbolic Demo Completed Successfully!")


if __name__ == "__main__":
    main()
