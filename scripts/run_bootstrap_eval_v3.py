"""Bootstrap eval v3 — Fix Wilcoxon signed-rank test.

Mudanças vs v2:

  1. WILCOXON CORRIGIDO: a raiz do problema era o torch.randperm() em
     build_regression_labels(), que embaralhava os pares de maneira diferente
     para cada modelo (o estado do RNG global diverge após treinamentos distintos).
     O fix está em link_prediction.py (remoção do shuffle) — agora ambos os modelos
     emitem preds/targets na mesma ordem determinística (sorted unique pairs do dict).
     Esta v3 simplesmente verifica np.allclose(tgn_targets, roland_targets) e roda
     o Wilcoxon diretamente, sem a gambiarra de argsort intra-dia da v2.

  2. DIAGNÓSTICO MELHORADO: logs detalhados de alinhamento antes do Wilcoxon.

  Tudo o mais (cache, LR por modelo, bootstrap pareado, CVaR) é idêntico à v2.
"""

import sys
import hashlib
import pickle
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

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

# ---------------------------------------------------------------------------
# Hiperparâmetros por modelo (idênticos à v2)
# ---------------------------------------------------------------------------

TGN_LR = 1e-3
TGN_USE_COSINE = False
TGN_PATIENCE = 5

BASELINE_LR = 1e-3
BASELINE_USE_COSINE = False
BASELINE_PATIENCE = 5


# ---------------------------------------------------------------------------
# Block Bootstrap
# ---------------------------------------------------------------------------

def _sharpe(arr: np.ndarray) -> float:
    """Sharpe proxy anualizado com std amostral (ddof=1)."""
    std = np.std(arr, ddof=1)
    return (float(np.mean(arr)) / std) * np.sqrt(252) if std > 1e-8 else 0.0


def block_bootstrap_metrics(
    returns: np.ndarray,
    block_size: int = 5,
    n_iterations: int = 20_000,
    seed: int = 42,
    cvar_alpha: float = 0.05,
) -> dict:
    """Distribuição empírica do Sharpe proxy e CVaR via block bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(returns)
    n_blocks = n // block_size + (1 if n % block_size != 0 else 0)

    sharpes = []
    cvars = []
    for _ in range(n_iterations):
        start_indices = rng.integers(0, n - block_size + 1, size=n_blocks)
        sampled = []
        for si in start_indices:
            sampled.extend(returns[si : si + block_size])
        sampled = np.array(sampled[:n])

        sharpes.append(_sharpe(sampled))

        cutoff = np.percentile(sampled, cvar_alpha * 100)
        tail = sampled[sampled <= cutoff]
        cvars.append(float(np.mean(tail)) if len(tail) > 0 else float(np.min(sampled)))

    return {"sharpes": np.array(sharpes), "cvars": np.array(cvars)}


def paired_block_bootstrap_diff(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    block_size: int = 5,
    n_iterations: int = 20_000,
    seed: int = 42,
) -> np.ndarray:
    """Distribuição bootstrap da diferença Sharpe_A − Sharpe_B usando MESMOS blocos."""
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
        sampled_a, sampled_b = [], []
        for si in start_indices:
            sampled_a.extend(returns_a[si : si + block_size])
            sampled_b.extend(returns_b[si : si + block_size])

        sa = np.array(sampled_a[:n])
        sb = np.array(sampled_b[:n])
        diffs.append(_sharpe(sa) - _sharpe(sb))

    return np.array(diffs)


# ---------------------------------------------------------------------------
# Cache de dados
# ---------------------------------------------------------------------------

def load_or_prepare_data(tickers, start, end, benchmark, config, data_config, logger):
    """Carrega prepared_data do cache se disponível, senão baixa e salva."""
    cache_key = hashlib.md5(
        f"{sorted(tickers)}{start}{end}{benchmark}".encode()
    ).hexdigest()[:10]
    cache_path = RESULTS_DIR / f"prepared_data_cache_{cache_key}.pkl"

    if cache_path.exists():
        logger.info("Carregando dados do cache: %s", cache_path)
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        logger.info("Cache carregado: %d datas com eventos.", len(data["sorted_dates"]))
    else:
        logger.info("Cache nao encontrado. Baixando dados (isso pode levar alguns minutos)...")
        data = prepare_data(tickers, start, end, benchmark, config, data_config, logger)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Dados salvos no cache: %s", cache_path)

    return data


# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------

def train_tgn(data, tickers, start, end, benchmark, epochs, logger) -> dict:
    """Treina TGN com seed=42 e condições do v0.7 (lr=1e-3 + flat)."""
    logger.info("TGN seed=42 lr=%.0e cosine=%s epochs=%d", TGN_LR, TGN_USE_COSINE, epochs)
    return train_link_prediction(
        tickers=tickers,
        start=start,
        end=end,
        benchmark=benchmark,
        num_epochs=epochs,
        lr=TGN_LR,
        mode="regression",
        model_variant="tgn",
        seed=42,
        prepared_data=data,
        use_cosine_schedule=TGN_USE_COSINE,
        early_stopping_patience=TGN_PATIENCE,
        weight_decay=1e-4,
    )


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_bootstrap_eval_v3(
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    model_variants: list = None,
    epochs: int = 20,
):
    if model_variants is None:
        model_variants = ["tgn", "roland", "gat_static"]

    logger = setup_logging("dyfo.bootstrap_eval_v3", log_to_file=False)
    logger.info("=" * 60)
    logger.info("Bootstrap Eval v3 — Wilcoxon fix + H4 robusto")
    logger.info("=" * 60)

    base_config = DyFOConfig()
    data_config = DataConfig(
        tickers=TICKERS_30, benchmark_ticker="SPY",
        start_date=start, end_date=end,
    )

    # --- 1. Dados (com cache) ---
    data = load_or_prepare_data(
        TICKERS_30, start, end, "SPY", base_config, data_config, logger
    )

    results = {}
    returns_dict = {}
    preds_dict = {}
    targets_dict = {}

    # --- 2. Treinamento ---
    for variant in model_variants:
        logger.info("-" * 50)
        logger.info("Treinando variante: %s", variant.upper())

        if variant == "tgn":
            test_metrics = train_tgn(
                data, TICKERS_30, start, end, "SPY", epochs, logger
            )
        else:
            test_metrics = train_link_prediction(
                tickers=TICKERS_30,
                start=start,
                end=end,
                benchmark="SPY",
                num_epochs=epochs,
                lr=BASELINE_LR,
                mode="regression",
                model_variant=variant,
                seed=42,
                prepared_data=data,
                use_cosine_schedule=BASELINE_USE_COSINE,
                early_stopping_patience=BASELINE_PATIENCE,
                weight_decay=1e-4,
            )

        results[variant] = {k: v for k, v in test_metrics.items() if not k.startswith("_")}
        returns_dict[variant] = np.array(test_metrics.get("_realized_returns", []))
        if "_all_preds" in test_metrics and "_all_targets" in test_metrics:
            preds_dict[variant] = test_metrics["_all_preds"].cpu().numpy()
            targets_dict[variant] = test_metrics["_all_targets"].cpu().numpy()

        r2 = results[variant].get("r_squared", float("nan"))
        spearman = results[variant].get("spearman", float("nan"))
        sharpe = results[variant].get("sharpe_proxy", float("nan"))
        logger.info(
            "%s → R²=%.4f  Spearman=%.4f  Sharpe=%.4f",
            variant.upper(), r2, spearman, sharpe,
        )

    # --- 3. Block Bootstrap ---
    logger.info("=" * 60)
    logger.info("Block Bootstrap (20 000 iterações, blocos de 5 dias)")
    logger.info("=" * 60)

    n_iters = 20_000
    block_size = 5
    bootstrap_sharpes = {}
    bootstrap_cvars = {}

    for variant, rets in returns_dict.items():
        if len(rets) == 0:
            logger.error("Sem retornos para %s. Pulando bootstrap.", variant)
            continue

        metrics = block_bootstrap_metrics(
            rets, block_size=block_size, n_iterations=n_iters, seed=42
        )
        sharpes = metrics["sharpes"]
        cvars = metrics["cvars"]
        bootstrap_sharpes[variant] = sharpes
        bootstrap_cvars[variant] = cvars

        ci_lo = np.percentile(sharpes, 2.5)
        ci_hi = np.percentile(sharpes, 97.5)
        logger.info(
            "%s  Sharpe obs=%.4f  Bootstrap mean=%.4f  95%% CI=[%.4f, %.4f]",
            variant.upper(),
            results[variant].get("sharpe_proxy", 0.0),
            float(np.mean(sharpes)),
            ci_lo, ci_hi,
        )
        logger.info(
            "%s  CVaR(5%%) obs=%.4f  Bootstrap mean=%.4f",
            variant.upper(),
            float(np.mean(rets[rets <= np.percentile(rets, 5)])),
            float(np.mean(cvars))
        )

    # --- 4. H4: TGN > ROLAND (bootstrap pareado) ---
    p_value = None
    p_value_centered = None
    p_value_cvar = None
    if "tgn" in returns_dict and "roland" in returns_dict:
        rets_tgn = returns_dict["tgn"]
        rets_roland = returns_dict["roland"]

        if len(rets_tgn) == 0 or len(rets_roland) == 0:
            logger.error("Retornos vazios para TGN ou ROLAND. Pulando H4.")
        else:
            d_obs = _sharpe(rets_tgn) - _sharpe(rets_roland)

            diff_b = paired_block_bootstrap_diff(
                rets_tgn, rets_roland,
                block_size=block_size, n_iterations=n_iters, seed=42,
            )

            p_value = float(np.mean(diff_b <= 0))
            p_value_centered = float(np.mean((diff_b - float(np.mean(diff_b))) >= d_obs))

            logger.info("=" * 60)
            logger.info("HIPOTESE H4 (FINANCEIRA): TGN > ROLAND")
            logger.info("  Sharpe observado  TGN:    %.4f", _sharpe(rets_tgn))
            logger.info("  Sharpe observado  ROLAND: %.4f", _sharpe(rets_roland))
            logger.info("  Diferenca observada d_obs: %.4f", d_obs)
            logger.info("  Bootstrap mean(diff):      %.4f  std=%.4f",
                        float(np.mean(diff_b)), float(np.std(diff_b, ddof=1)))
            logger.info("  p-valor direto   P(diff_b<=0)                = %.4f", p_value)
            logger.info("  p-valor centrado P(diff_b - mean >= d_obs)   = %.4f", p_value_centered)

            if p_value_centered < 0.05:
                logger.info(">>> H4 SUPORTADA (p_centrado < 0.05) [PASS]")
            elif p_value < 0.05:
                logger.info(">>> H4 SUPORTADA pelo p_direto mas nao pelo centrado — resultado marginal.")
            else:
                logger.info(">>> H4 NAO SUPORTADA SIGNIFICATIVAMENTE. [FAIL]")

        # CVaR
        if "tgn" in bootstrap_cvars and "roland" in bootstrap_cvars:
            tgn_c = bootstrap_cvars["tgn"]
            roland_c = bootstrap_cvars["roland"]
            p_value_cvar = float(np.mean(tgn_c <= roland_c))
            logger.info("P(TGN CVaR <= ROLAND CVaR) = %.4f", p_value_cvar)
            if p_value_cvar < 0.05:
                logger.info(">>> H4 CVaR SUPORTADA! (p < 0.05) [PASS]")
            else:
                logger.info(">>> H4 CVaR NAO SUPORTADA SIGNIFICATIVAMENTE.")

    if "tgn" in returns_dict and "gat_static" in returns_dict:
        rets_tgn = returns_dict["tgn"]
        rets_gat = returns_dict["gat_static"]
        if len(rets_tgn) > 0 and len(rets_gat) > 0:
            diff_gat = paired_block_bootstrap_diff(
                rets_tgn, rets_gat,
                block_size=block_size, n_iterations=n_iters, seed=42,
            )
            p_gat = float(np.mean(diff_gat <= 0))
            logger.info("P(TGN Sharpe <= GAT_STATIC) [direto] = %.4f", p_gat)

    # --- 4.5. Wilcoxon Signed-Rank Test (CORRIGIDO na v3) ---
    #
    # Pré-condição: build_regression_labels() NÃO aplica mais randperm.
    # Ambos os modelos emitem preds/targets na mesma ordem determinística
    # (sorted unique pairs, iteração estável do dict Python 3.7+).
    # Portanto, np.allclose(tgn_targets, roland_targets) deve ser True.
    #
    # H0: mediana(|err_TGN| - |err_ROLAND|) >= 0  (TGN não é melhor)
    # H1: mediana(|err_TGN| - |err_ROLAND|) < 0   (TGN erra menos)
    # ---------------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("HIPOTESE PREDITIVA: TGN > ROLAND (Menor Erro Absoluto — Wilcoxon)")
    p_value_wilcoxon = None

    if "tgn" in preds_dict and "roland" in preds_dict:
        tgn_preds = preds_dict["tgn"]
        tgn_targets = targets_dict["tgn"]
        roland_preds = preds_dict["roland"]
        roland_targets = targets_dict["roland"]

        logger.info("  Comprimentos: TGN=%d, ROLAND=%d", len(tgn_preds), len(roland_preds))

        if len(tgn_preds) != len(roland_preds):
            logger.warning(
                "  Wilcoxon ABORTADO: comprimentos diferentes (%d vs %d). "
                "Verifique se ambos os modelos usam os mesmos test_dates.",
                len(tgn_preds), len(roland_preds),
            )
        else:
            # Diagnóstico de alinhamento
            max_target_diff = float(np.max(np.abs(tgn_targets - roland_targets)))
            mean_target_diff = float(np.mean(np.abs(tgn_targets - roland_targets)))
            targets_match = np.allclose(tgn_targets, roland_targets, atol=1e-5)

            logger.info("  Alinhamento de targets:")
            logger.info("    max|tgn_target - roland_target| = %.6f", max_target_diff)
            logger.info("    mean|tgn_target - roland_target| = %.6f", mean_target_diff)
            logger.info("    np.allclose(atol=1e-5) = %s", targets_match)

            if not targets_match:
                # Diagnóstico adicional: quantos samples divergem?
                mismatched = np.sum(np.abs(tgn_targets - roland_targets) > 1e-5)
                logger.warning(
                    "  Wilcoxon ABORTADO: targets NAO alinhados! "
                    "%d/%d samples divergem (max_diff=%.4f). "
                    "Isso indica que build_regression_labels ainda está "
                    "shuffling ou os modelos viram dias diferentes.",
                    mismatched, len(tgn_targets), max_target_diff,
                )
            else:
                logger.info("  ✓ Targets perfeitamente alinhados — Wilcoxon válido.")

                abs_err_tgn = np.abs(tgn_preds - tgn_targets)
                abs_err_roland = np.abs(roland_preds - roland_targets)

                logger.info("  Erro Absoluto Medio TGN:    %.6f", float(np.mean(abs_err_tgn)))
                logger.info("  Erro Absoluto Medio ROLAND: %.6f", float(np.mean(abs_err_roland)))
                logger.info("  Erro Absoluto Mediano TGN:    %.6f", float(np.median(abs_err_tgn)))
                logger.info("  Erro Absoluto Mediano ROLAND: %.6f", float(np.median(abs_err_roland)))

                diff = abs_err_tgn - abs_err_roland
                logger.info("  mean(|err_TGN| - |err_ROLAND|) = %.6f", float(np.mean(diff)))
                logger.info("  median(|err_TGN| - |err_ROLAND|) = %.6f", float(np.median(diff)))
                logger.info("  P(TGN erra menos) = %.4f",
                            float(np.mean(abs_err_tgn < abs_err_roland)))

                try:
                    stat, p_value_wilcoxon = wilcoxon(
                        abs_err_tgn, abs_err_roland, alternative='less'
                    )
                    logger.info("  Wilcoxon statistic = %.4f", stat)
                    logger.info("  Wilcoxon p-value (TGN < ROLAND) = %.4e", p_value_wilcoxon)

                    if p_value_wilcoxon < 0.05:
                        logger.info(">>> H_PREDITIVA SUPORTADA! (p_wilcoxon < 0.05) [PASS]")
                    else:
                        logger.info(">>> H_PREDITIVA NAO SUPORTADA. (p=%.4e) [FAIL]",
                                    p_value_wilcoxon)
                except Exception as e:
                    logger.error("  Wilcoxon falhou com exceção: %s", e)
    else:
        logger.warning("  Wilcoxon pulado: preds de TGN ou ROLAND não disponíveis.")

    # --- 5. Resumo de R² ---
    logger.info("=" * 60)
    logger.info("RESUMO R²")
    for v, m in results.items():
        r2 = m.get("r_squared", float("nan"))
        flag = " [>80% TARGET MET]" if v == "tgn" and r2 >= 0.80 else ""
        logger.info("  %-12s R²=%.4f%s", v.upper(), r2, flag)

    # --- 6. Output ---
    out_dir = RESULTS_DIR / f"bootstrap_eval_v3_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": "v3",
        "p_value_tgn_vs_roland_direct": p_value,
        "p_value_tgn_vs_roland_centered": p_value_centered,
        "p_value_cvar": p_value_cvar,
        "p_value_wilcoxon": p_value_wilcoxon,
        "h4_supported": bool(p_value_centered is not None and p_value_centered < 0.05),
        "h4_predictive_supported": bool(p_value_wilcoxon is not None and p_value_wilcoxon < 0.05),
        "metrics": results,
        "bootstrap_mean_sharpes": {k: float(np.mean(v)) for k, v in bootstrap_sharpes.items()},
        "bootstrap_ci_2.5": {k: float(np.percentile(v, 2.5)) for k, v in bootstrap_sharpes.items()},
        "bootstrap_ci_97.5": {k: float(np.percentile(v, 97.5)) for k, v in bootstrap_sharpes.items()},
        "config": {
            "tgn_lr": TGN_LR,
            "tgn_use_cosine_schedule": TGN_USE_COSINE,
            "tgn_patience": TGN_PATIENCE,
            "baseline_lr": BASELINE_LR,
            "baseline_use_cosine_schedule": BASELINE_USE_COSINE,
            "baseline_patience": BASELINE_PATIENCE,
            "epochs": epochs,
            "bootstrap_n_iterations": n_iters,
            "bootstrap_block_size": block_size,
        },
    }
    out_path = out_dir / "bootstrap_summary_v3.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Resumo salvo em: %s", out_path)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bootstrap Eval v3 — Wilcoxon fix")
    parser.add_argument("--epochs", type=int, default=20, help="Epocas de treinamento (default: 20)")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument(
        "--variants", nargs="+", default=["tgn", "roland", "gat_static"],
        help="Variantes a treinar (default: tgn roland gat_static)",
    )
    args = parser.parse_args()

    run_bootstrap_eval_v3(
        start=args.start,
        end=args.end,
        model_variants=args.variants,
        epochs=args.epochs,
    )
