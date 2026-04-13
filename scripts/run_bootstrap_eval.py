"""Script to evaluate H4 via Out-of-Sample Block Bootstrap.

This script replaces the expensive multi-window walk-forward validation.
It trains TGN, ROLAND, and GAT_STATIC once on a single historical split
(e.g., Train: 60%, Val: 20%, Test: 20%).
After training, it extracts the out-of-sample portfolio returns
(_realized_returns) from the Test period and performs Block Bootstrapping
to generate empirical p-values for H4 (TGN > ROLAND).
"""

import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dyfo.config import DataConfig, DyFOConfig
from dyfo.logging_utils import RESULTS_DIR, setup_logging
from scripts.train_link_prediction import prepare_data, train_link_prediction

TICKERS_30 = [
    "AAPL", "MSFT", "NVDA", "AVGO", "CRM",
    "JPM", "GS", "MA", "BRK-B",
    "JNJ", "UNH", "LLY",
    "AMZN", "TSLA", "HD",
    "PG", "KO",
    "XOM", "CVX",
    "CAT", "BA", "RTX",
    "META", "GOOGL", "DIS",
    "LIN", "APD",
    "NEE", "DUK",
    "PLD",
]

def _sharpe(arr: np.ndarray) -> float:
    """Sharpe proxy anualizado com std amostral (ddof=1)."""
    std = np.std(arr, ddof=1)
    return (float(np.mean(arr)) / std) * np.sqrt(252) if std > 1e-8 else 0.0


def block_bootstrap_sharpe(returns: np.ndarray, block_size: int = 5, n_iterations: int = 10000, seed: int = 42):
    """Computes distribution of Sharpe proxy using block bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(returns)
    n_blocks = n // block_size + (1 if n % block_size != 0 else 0)

    sharpes = []
    for _ in range(n_iterations):
        start_indices = rng.integers(0, n - block_size + 1, size=n_blocks)
        sampled_returns = []
        for start_idx in start_indices:
            sampled_returns.extend(returns[start_idx:start_idx + block_size])
        sampled_returns = np.array(sampled_returns[:n])
        sharpes.append(_sharpe(sampled_returns))

    return np.array(sharpes)


def paired_block_bootstrap_diff(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    block_size: int = 5,
    n_iterations: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """Distribuição bootstrap da diferença Sharpe_A − Sharpe_B usando MESMOS blocos.

    Um único RNG compartilhado garante que cada iteração avalia A e B sobre o
    mesmo período reamostrado — pareamento correto para comparação de estratégias.

    Raises:
        ValueError: Se as séries tiverem comprimentos diferentes.
    """
    if len(returns_a) != len(returns_b):
        raise ValueError(
            f"Séries com comprimentos diferentes: {len(returns_a)} vs {len(returns_b)}. "
            "O pareamento exige séries do mesmo período de teste."
        )
    rng = np.random.default_rng(seed)
    n = len(returns_a)
    n_blocks = n // block_size + (1 if n % block_size != 0 else 0)

    diffs = []
    for _ in range(n_iterations):
        start_indices = rng.integers(0, n - block_size + 1, size=n_blocks)
        sa, sb = [], []
        for si in start_indices:
            sa.extend(returns_a[si : si + block_size])
            sb.extend(returns_b[si : si + block_size])
        diffs.append(_sharpe(np.array(sa[:n])) - _sharpe(np.array(sb[:n])))

    return np.array(diffs)

def run_bootstrap_eval(
    start="2020-01-01",
    end="2024-12-31",
    model_variants=["tgn", "roland", "gat_static"],
    epochs=10
):
    logger = setup_logging("dyfo.bootstrap_eval", log_to_file=False)
    logger.info("Starting Bootstrap Evaluation for H4")

    config = DyFOConfig()
    data_config = DataConfig(tickers=TICKERS_30, benchmark_ticker="SPY", start_date=start, end_date=end)

    logger.info("Preparing data...")
    data = prepare_data(TICKERS_30, start, end, "SPY", config, data_config, logger)

    results = {}
    returns_dict = {}

    for variant in model_variants:
        logger.info(f"Training variant {variant}...")
        test_metrics = train_link_prediction(
            tickers=TICKERS_30,
            start=start,
            end=end,
            benchmark="SPY",
            num_epochs=epochs,
            lr=2e-4,
            mode="regression",
            model_variant=variant,
            seed=42,
            prepared_data=data,
        )
        results[variant] = {k: v for k, v in test_metrics.items() if not k.startswith("_")}

        # Save realized returns
        # _realized_returns contains list of daily GMV returns computed in run_split for "test"
        ret = test_metrics.get("_realized_returns", [])
        returns_dict[variant] = np.array(ret)

    logger.info("Training complete. Starting Block Bootstrap...")

    # Bootstrap
    n_iters = 10000
    block_size = 5
    bootstrap_sharpes = {}

    for variant, rets in returns_dict.items():
        if len(rets) == 0:
            logger.error(f"No returns found for {variant}. Skipping bootstrap.")
            continue

        sharpes = block_bootstrap_sharpe(rets, block_size=block_size, n_iterations=n_iters, seed=42)
        bootstrap_sharpes[variant] = sharpes

        ci_lower = np.percentile(sharpes, 2.5)
        ci_upper = np.percentile(sharpes, 97.5)
        avg_sharpe = np.mean(sharpes)
        logger.info(f"{variant.upper()} - Sharpe: {results[variant].get('sharpe_proxy', 0):.4f} | Bootstrap Mean: {avg_sharpe:.4f} | 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # H0: Sharpe_TGN ≤ Sharpe_ROLAND
    # H1: Sharpe_TGN > Sharpe_ROLAND  (H4)
    #
    # Bootstrap pareado: mesmos blocos em cada iteração → diff por iteração.
    # p_direto   = P(diff_b ≤ 0)
    # p_centrado = P(diff_b − mean(diff_b) ≥ d_obs)  — calibrado sob H0
    p_value = None
    p_value_centered = None
    if "tgn" in returns_dict and "roland" in returns_dict:
        rets_tgn = returns_dict["tgn"]
        rets_roland = returns_dict["roland"]

        d_obs = _sharpe(rets_tgn) - _sharpe(rets_roland)
        diff_b = paired_block_bootstrap_diff(
            rets_tgn, rets_roland,
            block_size=block_size, n_iterations=n_iters, seed=42,
        )
        p_value = float(np.mean(diff_b <= 0))
        p_value_centered = float(np.mean((diff_b - float(np.mean(diff_b))) >= d_obs))

        logger.info("=" * 50)
        logger.info("HYPOTHESIS H4 VALIDATION")
        logger.info("=" * 50)
        logger.info(f"  Sharpe observado TGN:    {_sharpe(rets_tgn):.4f}")
        logger.info(f"  Sharpe observado ROLAND: {_sharpe(rets_roland):.4f}")
        logger.info(f"  Diferença observada d_obs: {d_obs:.4f}")
        logger.info(f"  p-valor direto   P(diff_b<=0)              = {p_value:.4f}")
        logger.info(f"  p-valor centrado P(diff_b-mean >= d_obs)   = {p_value_centered:.4f}")
        if p_value_centered < 0.05:
            logger.info(">>> HYPOTHESIS H4 SUPPORTED! (p_centrado < 0.05) [PASS]")
        elif p_value < 0.05:
            logger.info(">>> H4 suportada pelo p_direto mas não pelo centrado — resultado marginal.")
        else:
            logger.info(">>> HYPOTHESIS H4 NOT SIGNIFICANTLY SUPPORTED. [FAIL]")

    if "tgn" in returns_dict and "gat_static" in returns_dict:
        rets_tgn = returns_dict["tgn"]
        rets_gat = returns_dict["gat_static"]
        if len(rets_tgn) > 0 and len(rets_gat) > 0:
            diff_gat = paired_block_bootstrap_diff(
                rets_tgn, rets_gat,
                block_size=block_size, n_iterations=n_iters, seed=42,
            )
            logger.info(f"P(TGN Sharpe <= GAT_STATIC) [direto] = {float(np.mean(diff_gat <= 0)):.4f}")

    # Output to File
    out_dir = RESULTS_DIR / f"bootstrap_eval_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "bootstrap_summary.json", "w", encoding="utf-8") as f:
        json_res = {
            # p_value_direct: P(diff_bootstrap <= 0)
            "p_value_tgn_vs_roland_direct": float(p_value) if p_value is not None else None,
            # p_value_centered: calibrado sob H0 (mais rigoroso)
            "p_value_tgn_vs_roland_centered": float(p_value_centered) if p_value_centered is not None else None,
            "h4_supported": bool(p_value_centered is not None and p_value_centered < 0.05),
            "metrics": results,
            "bootstrap_mean_sharpes": {k: float(np.mean(v)) for k, v in bootstrap_sharpes.items()},
            "bootstrap_ci_2.5": {k: float(np.percentile(v, 2.5)) for k, v in bootstrap_sharpes.items()},
            "bootstrap_ci_97.5": {k: float(np.percentile(v, 97.5)) for k, v in bootstrap_sharpes.items()},
        }
        json.dump(json_res, f, indent=2)

    logger.info(f"Summary saved to {out_dir / 'bootstrap_summary.json'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    run_bootstrap_eval(epochs=args.epochs)
