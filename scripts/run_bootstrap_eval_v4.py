"""Bootstrap eval v3 — Robust statistical testing suite.

Mudanças vs v2:

  1. WILCOXON CORRIGIDO: remoção do torch.randperm() em build_regression_labels().

  2. DIEBOLD-MARIANO TEST: Teste padrão em econometria de previsão (Diebold & Mariano,
     1995) com estimação HAC Newey-West para autocorrelação serial nos erros.
     Complementa o Wilcoxon com inferência robusta à dependência temporal.

  3. EFFECT SIZES: Cohen's d para bootstrap Sharpe diff e DM test. Rank-biserial
     correlation para o Wilcoxon. ASA (2016) recomenda reportar junto com p-valores.

  4. HOLM-BONFERRONI: Correção para testes múltiplos. Com 7+ testes simultâneos,
     o FWER sem correção é ~30%. Holm-Bonferroni controla o FWER em α=0.05.

  5. CI PARA SHARPE DIFFERENCE: Percentis 2.5/97.5 da distribuição bootstrap pareada.

  6. CVaR PAREADO + INDEPENDENTE: Ambos os métodos, para robustez total.

  7. WILCOXON COMPLETO: TGN vs ROLAND + TGN vs GAT_STATIC.
"""

import sys
import hashlib
import pickle
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, norm, rankdata

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

N_PAIRS = 435  # C(30, 2) — número de pares únicos

# ---------------------------------------------------------------------------
# Hiperparâmetros por modelo
# ---------------------------------------------------------------------------

TGN_LR = 1e-3
TGN_USE_COSINE = False
TGN_PATIENCE = 5

BASELINE_LR = 1e-3
BASELINE_USE_COSINE = False
BASELINE_PATIENCE = 5


# ---------------------------------------------------------------------------
# Utilidades estatísticas
# ---------------------------------------------------------------------------

def _sharpe(arr: np.ndarray) -> float:
    """Sharpe proxy anualizado com std amostral (ddof=1)."""
    std = np.std(arr, ddof=1)
    return (float(np.mean(arr)) / std) * np.sqrt(252) if std > 1e-8 else 0.0


def _cvar(arr: np.ndarray, alpha: float = 0.05) -> float:
    """CVaR (Expected Shortfall) at alpha level."""
    cutoff = np.percentile(arr, alpha * 100)
    tail = arr[arr <= cutoff]
    return float(np.mean(tail)) if len(tail) > 0 else float(np.min(arr))


def cohens_d(diff: np.ndarray) -> float:
    """Cohen's d effect size from a distribution of differences."""
    std = np.std(diff, ddof=1)
    return float(np.mean(diff) / std) if std > 1e-10 else 0.0


def rank_biserial_from_wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (effect size for Wilcoxon).

    Interpretation: r > 0 means x < y more often (TGN errors < ROLAND errors).
    |r| < 0.1 = negligible, 0.1–0.3 = small, 0.3–0.5 = medium, > 0.5 = large.
    """
    d = x - y
    nonzero_mask = d != 0
    d_nz = d[nonzero_mask]
    if len(d_nz) == 0:
        return 0.0

    ranks = rankdata(np.abs(d_nz))
    w_plus = float(np.sum(ranks[d_nz > 0]))   # TGN erra MAIS
    w_minus = float(np.sum(ranks[d_nz < 0]))   # TGN erra MENOS
    total = w_plus + w_minus
    if total == 0:
        return 0.0
    # Positive r = TGN erra menos (favorável ao TGN)
    return (w_minus - w_plus) / total


def newey_west_variance(d: np.ndarray, max_lags: int = None) -> float:
    """Newey-West HAC estimator of the long-run variance of d.

    Uses Bartlett kernel weights: w(k) = 1 - k/(L+1).
    """
    T = len(d)
    if max_lags is None:
        max_lags = int(np.floor(T ** (1.0 / 3.0)))

    d_demeaned = d - np.mean(d)

    # Autocovariances
    gamma = np.zeros(max_lags + 1)
    for k in range(max_lags + 1):
        gamma[k] = np.dot(d_demeaned[:T - k], d_demeaned[k:]) / T

    # HAC variance = gamma(0) + 2 * sum_{k=1}^{L} (1 - k/(L+1)) * gamma(k)
    var_hac = gamma[0]
    for k in range(1, max_lags + 1):
        weight = 1.0 - k / (max_lags + 1)
        var_hac += 2.0 * weight * gamma[k]

    return max(var_hac, 1e-20)  # floor to avoid division by zero


def diebold_mariano_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    loss: str = "mse",
    alternative: str = "less",
) -> dict:
    """Diebold-Mariano test (1995) com HAC Newey-West.

    H0: E[L(e_a) - L(e_b)] = 0  (equal predictive accuracy)

    Args:
        errors_a: Prediction errors of model A (TGN) — one per day.
        errors_b: Prediction errors of model B (baseline) — one per day.
        loss: "mse" (squared error) or "mae" (absolute error).
        alternative: "less" (A better than B), "greater", or "two-sided".

    Returns:
        dict with dm_statistic, p_value, mean_loss_diff, effect_size_d.
    """
    if loss == "mse":
        loss_a = errors_a ** 2
        loss_b = errors_b ** 2
    elif loss == "mae":
        loss_a = np.abs(errors_a)
        loss_b = np.abs(errors_b)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    d = loss_a - loss_b  # negative = A is better
    T = len(d)
    d_bar = float(np.mean(d))

    var_hac = newey_west_variance(d)
    se = np.sqrt(var_hac / T)
    dm_stat = d_bar / se if se > 1e-10 else 0.0

    if alternative == "less":
        p_val = float(norm.cdf(dm_stat))
    elif alternative == "greater":
        p_val = float(1 - norm.cdf(dm_stat))
    else:  # two-sided
        p_val = float(2 * norm.cdf(-abs(dm_stat)))

    return {
        "dm_statistic": dm_stat,
        "p_value": p_val,
        "mean_loss_diff": d_bar,
        "effect_size_d": cohens_d(d),
        "n_days": T,
        "max_lags_hac": int(np.floor(T ** (1.0 / 3.0))),
    }


def holm_bonferroni(p_values: dict) -> dict:
    """Holm-Bonferroni correction for multiple testing.

    More powerful than classic Bonferroni while still controlling FWER.
    Returns corrected p-values and pass/fail at α=0.05.
    """
    # Filter out None values
    valid = {k: v for k, v in p_values.items() if v is not None}
    if not valid:
        return {}

    sorted_tests = sorted(valid.items(), key=lambda x: x[1])
    m = len(sorted_tests)
    corrected = {}
    max_so_far = 0.0
    for rank, (name, p) in enumerate(sorted_tests):
        # Holm: multiply by (m - rank), enforce monotonicity
        adj_p = min(1.0, p * (m - rank))
        adj_p = max(adj_p, max_so_far)  # enforce monotonicity
        max_so_far = adj_p
        corrected[name] = {
            "original_p": p,
            "corrected_p": adj_p,
            "significant_at_0.05": adj_p < 0.05,
        }

    return corrected


# ---------------------------------------------------------------------------
# Block Bootstrap
# ---------------------------------------------------------------------------

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
        cvars.append(_cvar(sampled, cvar_alpha))

    return {"sharpes": np.array(sharpes), "cvars": np.array(cvars)}


def paired_block_bootstrap_multi(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    block_size: int = 5,
    n_iterations: int = 20_000,
    seed: int = 42,
    cvar_alpha: float = 0.05,
) -> dict:
    """Bootstrap pareado: diferenças de Sharpe E CVaR usando MESMOS blocos.

    Retorna distribuições de:
      - sharpe_diffs: Sharpe_A − Sharpe_B por iteração
      - cvar_diffs: CVaR_A − CVaR_B por iteração (mais negativo = A tem menor tail risk)
    """
    if len(returns_a) != len(returns_b):
        raise ValueError(
            f"Séries com comprimentos diferentes: {len(returns_a)} vs {len(returns_b)}."
        )

    rng = np.random.default_rng(seed)
    n = len(returns_a)
    n_blocks = n // block_size + (1 if n % block_size != 0 else 0)

    sharpe_diffs = []
    cvar_diffs = []
    for _ in range(n_iterations):
        start_indices = rng.integers(0, n - block_size + 1, size=n_blocks)
        sampled_a, sampled_b = [], []
        for si in start_indices:
            sampled_a.extend(returns_a[si : si + block_size])
            sampled_b.extend(returns_b[si : si + block_size])

        sa = np.array(sampled_a[:n])
        sb = np.array(sampled_b[:n])

        sharpe_diffs.append(_sharpe(sa) - _sharpe(sb))
        cvar_diffs.append(_cvar(sa, cvar_alpha) - _cvar(sb, cvar_alpha))

    return {
        "sharpe_diffs": np.array(sharpe_diffs),
        "cvar_diffs": np.array(cvar_diffs),
    }


# ---------------------------------------------------------------------------
# Wilcoxon helper (with effect size)
# ---------------------------------------------------------------------------

def run_wilcoxon_test(
    preds_a: np.ndarray,
    targets_a: np.ndarray,
    preds_b: np.ndarray,
    targets_b: np.ndarray,
    name_a: str,
    name_b: str,
    logger,
) -> dict:
    """Run Wilcoxon signed-rank test comparing |err_A| vs |err_B|.

    Returns dict with p_value, statistic, effect_size_r, or None if alignment fails.
    """
    result = {"name": f"{name_a}_vs_{name_b}", "p_value": None, "statistic": None,
              "effect_size_r": None, "mean_err_a": None, "mean_err_b": None}

    if len(preds_a) != len(preds_b):
        logger.warning("  Wilcoxon %s vs %s: comprimentos diferentes (%d vs %d).",
                        name_a, name_b, len(preds_a), len(preds_b))
        return result

    targets_match = np.allclose(targets_a, targets_b, atol=1e-5)
    max_diff = float(np.max(np.abs(targets_a - targets_b)))

    if not targets_match:
        mismatched = int(np.sum(np.abs(targets_a - targets_b) > 1e-5))
        logger.warning(
            "  Wilcoxon %s vs %s: targets NAO alinhados! %d/%d divergem (max=%.4f).",
            name_a, name_b, mismatched, len(targets_a), max_diff,
        )
        return result

    abs_err_a = np.abs(preds_a - targets_a)
    abs_err_b = np.abs(preds_b - targets_b)

    result["mean_err_a"] = float(np.mean(abs_err_a))
    result["mean_err_b"] = float(np.mean(abs_err_b))

    # Effect size
    r_rb = rank_biserial_from_wilcoxon(abs_err_a, abs_err_b)
    result["effect_size_r"] = r_rb

    try:
        stat, p_val = wilcoxon(abs_err_a, abs_err_b, alternative='less')
        result["statistic"] = float(stat)
        result["p_value"] = float(p_val)

        logger.info("  Wilcoxon %s vs %s:", name_a, name_b)
        logger.info("    Targets alinhados: max_diff=%.2e ✓", max_diff)
        logger.info("    MAE %s: %.6f  |  MAE %s: %.6f", name_a, result["mean_err_a"],
                     name_b, result["mean_err_b"])
        logger.info("    statistic=%.2f  p-value=%.4e", stat, p_val)
        logger.info("    rank-biserial r=%.4f (%s)",
                     r_rb, _interpret_effect_r(r_rb))

        if p_val < 0.05:
            logger.info("    >>> %s ERRA SIGNIFICATIVAMENTE MENOS que %s [PASS]",
                         name_a, name_b)
        else:
            logger.info("    >>> Diferença NÃO significativa [FAIL]")

    except Exception as e:
        logger.error("  Wilcoxon %s vs %s: exceção: %s", name_a, name_b, e)

    return result


def _interpret_effect_r(r: float) -> str:
    """Interpret rank-biserial effect size magnitude."""
    ar = abs(r)
    if ar < 0.1:
        return "negligível"
    elif ar < 0.3:
        return "pequeno"
    elif ar < 0.5:
        return "médio"
    else:
        return "grande"


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size magnitude."""
    ad = abs(d)
    if ad < 0.2:
        return "negligível"
    elif ad < 0.5:
        return "pequeno"
    elif ad < 0.8:
        return "médio"
    else:
        return "grande"


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
    """Treina TGN com seed=42."""
    logger.info("TGN seed=42 lr=%.0e cosine=%s epochs=%d", TGN_LR, TGN_USE_COSINE, epochs)
    return train_link_prediction(
        tickers=tickers, start=start, end=end, benchmark=benchmark,
        num_epochs=epochs, lr=TGN_LR, mode="regression", model_variant="tgn",
        seed=42, prepared_data=data, use_cosine_schedule=TGN_USE_COSINE,
        early_stopping_patience=TGN_PATIENCE, weight_decay=1e-4,
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
    logger.info("Bootstrap Eval v3 — Robust Statistical Testing Suite")
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
                tickers=TICKERS_30, start=start, end=end, benchmark="SPY",
                num_epochs=epochs, lr=BASELINE_LR, mode="regression",
                model_variant=variant, seed=42, prepared_data=data,
                use_cosine_schedule=BASELINE_USE_COSINE,
                early_stopping_patience=BASELINE_PATIENCE, weight_decay=1e-4,
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

    # --- Collector for all p-values (Holm-Bonferroni at the end) ---
    all_p_values = {}

    # --- 3. Block Bootstrap (individual) ---
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
        bootstrap_sharpes[variant] = metrics["sharpes"]
        bootstrap_cvars[variant] = metrics["cvars"]

        ci_lo = np.percentile(metrics["sharpes"], 2.5)
        ci_hi = np.percentile(metrics["sharpes"], 97.5)
        logger.info(
            "%s  Sharpe obs=%.4f  Bootstrap mean=%.4f  95%% CI=[%.4f, %.4f]",
            variant.upper(),
            results[variant].get("sharpe_proxy", 0.0),
            float(np.mean(metrics["sharpes"])), ci_lo, ci_hi,
        )
        logger.info(
            "%s  CVaR(5%%) obs=%.4f  Bootstrap mean=%.4f",
            variant.upper(),
            _cvar(rets), float(np.mean(metrics["cvars"])),
        )

    # --- 4. H4: TGN > ROLAND (bootstrap pareado — Sharpe + CVaR) ---
    sharpe_diff_results = {}
    cvar_paired_results = {}

    if "tgn" in returns_dict and "roland" in returns_dict:
        rets_tgn = returns_dict["tgn"]
        rets_roland = returns_dict["roland"]

        if len(rets_tgn) == 0 or len(rets_roland) == 0:
            logger.error("Retornos vazios para TGN ou ROLAND. Pulando H4.")
        else:
            logger.info("=" * 60)
            logger.info("HIPOTESE H4 (FINANCEIRA): TGN > ROLAND")

            # --- Paired bootstrap: Sharpe + CVaR together ---
            paired = paired_block_bootstrap_multi(
                rets_tgn, rets_roland,
                block_size=block_size, n_iterations=n_iters, seed=42,
            )
            diff_b = paired["sharpe_diffs"]
            cvar_diff_b = paired["cvar_diffs"]

            d_obs = _sharpe(rets_tgn) - _sharpe(rets_roland)

            # Sharpe p-values
            p_direto = float(np.mean(diff_b <= 0))
            p_centrado = float(np.mean((diff_b - float(np.mean(diff_b))) >= d_obs))

            # Sharpe diff CI
            ci_diff_lo = float(np.percentile(diff_b, 2.5))
            ci_diff_hi = float(np.percentile(diff_b, 97.5))

            # Effect size
            d_cohen = cohens_d(diff_b)

            sharpe_diff_results["tgn_vs_roland"] = {
                "d_obs": d_obs,
                "p_direct": p_direto,
                "p_centered": p_centrado,
                "ci_2.5": ci_diff_lo,
                "ci_97.5": ci_diff_hi,
                "cohens_d": d_cohen,
                "bootstrap_mean": float(np.mean(diff_b)),
                "bootstrap_std": float(np.std(diff_b, ddof=1)),
            }

            all_p_values["H4_sharpe_direct_tgn_vs_roland"] = p_direto
            all_p_values["H4_sharpe_centered_tgn_vs_roland"] = p_centrado

            logger.info("  Sharpe observado  TGN: %.4f  ROLAND: %.4f", _sharpe(rets_tgn), _sharpe(rets_roland))
            logger.info("  Diferença observada: %.4f", d_obs)
            logger.info("  Bootstrap: mean=%.4f  std=%.4f  95%% CI=[%.4f, %.4f]",
                         float(np.mean(diff_b)), float(np.std(diff_b, ddof=1)),
                         ci_diff_lo, ci_diff_hi)
            logger.info("  p_direto=%.4f  p_centrado=%.4f", p_direto, p_centrado)
            logger.info("  Cohen's d=%.4f (%s)", d_cohen, _interpret_cohens_d(d_cohen))

            if p_centrado < 0.05:
                logger.info(">>> H4 Sharpe SUPORTADA (p_centrado < 0.05) [PASS]")
            elif p_direto < 0.05:
                logger.info(">>> H4 Sharpe SUPORTADA pelo p_direto; centrado marginal.")
            else:
                logger.info(">>> H4 Sharpe NAO SUPORTADA. [FAIL]")

            # --- CVaR paired ---
            p_cvar_paired = float(np.mean(cvar_diff_b >= 0))
            # Se TGN tem menor (mais negativo) CVaR, diff = CVaR_TGN - CVaR_ROLAND < 0
            # Queremos P(diff >= 0) como p-valor (fração onde TGN NAO é melhor)
            # Mas a interpretação depende: CVaR mais negativo = pior tail loss
            # Na verdade, queremos teste one-sided: TGN CVaR <= ROLAND CVaR (TGN perde menos na cauda)
            # diff < 0 → TGN tem CVaR mais negativo (pior) → depende da convenção
            # CVaR < 0 em retornos diários → mais negativo = pior
            # P(CVaR_TGN <= CVaR_ROLAND) = P(diff <= 0)
            p_cvar_paired = float(np.mean(cvar_diff_b <= 0))

            # CVaR independente (como v2)
            if "tgn" in bootstrap_cvars and "roland" in bootstrap_cvars:
                p_cvar_indep = float(np.mean(bootstrap_cvars["tgn"] <= bootstrap_cvars["roland"]))
            else:
                p_cvar_indep = None

            cvar_paired_results["tgn_vs_roland"] = {
                "p_paired": p_cvar_paired,
                "p_independent": p_cvar_indep,
                "cvar_obs_tgn": _cvar(rets_tgn),
                "cvar_obs_roland": _cvar(rets_roland),
            }

            all_p_values["H4_cvar_paired_tgn_vs_roland"] = p_cvar_paired
            if p_cvar_indep is not None:
                all_p_values["H4_cvar_indep_tgn_vs_roland"] = p_cvar_indep

            logger.info("  CVaR obs TGN: %.6f  ROLAND: %.6f", _cvar(rets_tgn), _cvar(rets_roland))
            logger.info("  P(TGN CVaR <= ROLAND CVaR) pareado=%.4f  independente=%.4f",
                         p_cvar_paired, p_cvar_indep if p_cvar_indep is not None else float("nan"))
            if p_cvar_paired < 0.05:
                logger.info(">>> H4 CVaR SUPORTADA (pareado, p < 0.05) [PASS]")
            else:
                logger.info(">>> H4 CVaR NAO SUPORTADA (pareado). [FAIL]")

    # --- 4b. TGN vs GAT_STATIC (bootstrap pareado completo) ---
    if "tgn" in returns_dict and "gat_static" in returns_dict:
        rets_tgn = returns_dict["tgn"]
        rets_gat = returns_dict["gat_static"]
        if len(rets_tgn) > 0 and len(rets_gat) > 0:
            logger.info("-" * 40)
            logger.info("TGN vs GAT_STATIC (bootstrap pareado)")

            paired_gat = paired_block_bootstrap_multi(
                rets_tgn, rets_gat,
                block_size=block_size, n_iterations=n_iters, seed=42,
            )
            diff_gat = paired_gat["sharpe_diffs"]
            d_obs_gat = _sharpe(rets_tgn) - _sharpe(rets_gat)

            p_gat_direto = float(np.mean(diff_gat <= 0))
            p_gat_centrado = float(np.mean((diff_gat - float(np.mean(diff_gat))) >= d_obs_gat))
            ci_gat_lo = float(np.percentile(diff_gat, 2.5))
            ci_gat_hi = float(np.percentile(diff_gat, 97.5))
            d_cohen_gat = cohens_d(diff_gat)

            sharpe_diff_results["tgn_vs_gat_static"] = {
                "d_obs": d_obs_gat,
                "p_direct": p_gat_direto,
                "p_centered": p_gat_centrado,
                "ci_2.5": ci_gat_lo,
                "ci_97.5": ci_gat_hi,
                "cohens_d": d_cohen_gat,
            }

            all_p_values["sharpe_direct_tgn_vs_gat_static"] = p_gat_direto

            logger.info("  Sharpe diff obs: %.4f  p_direto=%.4f  CI=[%.4f, %.4f]",
                         d_obs_gat, p_gat_direto, ci_gat_lo, ci_gat_hi)
            logger.info("  Cohen's d=%.4f (%s)", d_cohen_gat, _interpret_cohens_d(d_cohen_gat))

    # --- 5. Wilcoxon Signed-Rank (todas as comparações) ---
    logger.info("=" * 60)
    logger.info("TESTES PREDITIVOS — Wilcoxon Signed-Rank")
    wilcoxon_results = {}

    # TGN vs ROLAND
    if "tgn" in preds_dict and "roland" in preds_dict:
        w_res = run_wilcoxon_test(
            preds_dict["tgn"], targets_dict["tgn"],
            preds_dict["roland"], targets_dict["roland"],
            "TGN", "ROLAND", logger,
        )
        wilcoxon_results["tgn_vs_roland"] = w_res
        if w_res["p_value"] is not None:
            all_p_values["wilcoxon_tgn_vs_roland"] = w_res["p_value"]

    # TGN vs GAT_STATIC
    if "tgn" in preds_dict and "gat_static" in preds_dict:
        w_res_gat = run_wilcoxon_test(
            preds_dict["tgn"], targets_dict["tgn"],
            preds_dict["gat_static"], targets_dict["gat_static"],
            "TGN", "GAT_STATIC", logger,
        )
        wilcoxon_results["tgn_vs_gat_static"] = w_res_gat
        if w_res_gat["p_value"] is not None:
            all_p_values["wilcoxon_tgn_vs_gat_static"] = w_res_gat["p_value"]

    # --- 6. Diebold-Mariano Test ---
    logger.info("=" * 60)
    logger.info("TESTES PREDITIVOS — Diebold-Mariano (HAC Newey-West)")

    dm_results = {}

    for name_b, variant_b in [("ROLAND", "roland"), ("GAT_STATIC", "gat_static")]:
        if "tgn" not in preds_dict or variant_b not in preds_dict:
            continue

        tgn_p = preds_dict["tgn"]
        tgn_t = targets_dict["tgn"]
        b_p = preds_dict[variant_b]
        b_t = targets_dict[variant_b]

        if len(tgn_p) != len(b_p):
            logger.warning("  DM TGN vs %s: comprimentos diferentes.", name_b)
            continue
        if not np.allclose(tgn_t, b_t, atol=1e-5):
            logger.warning("  DM TGN vs %s: targets não alinhados.", name_b)
            continue

        # Aggregate to daily level: reshape to [D, N_PAIRS], compute daily MAE
        n_total = len(tgn_p)
        n_pairs = N_PAIRS
        if n_total % n_pairs != 0:
            # Fallback: try to infer n_pairs
            logger.warning("  DM: n_total=%d não é múltiplo de %d. Tentando inferir.", n_total, n_pairs)
            # Use the actual number of pairs per day from the first day
            n_pairs = n_total  # treat as single day — not ideal
            n_days = 1
        else:
            n_days = n_total // n_pairs

        if n_days < 10:
            logger.warning("  DM TGN vs %s: apenas %d dias — insuficiente para DM.", name_b, n_days)
            continue

        # Reshape and compute daily errors
        tgn_err = np.abs(tgn_p - tgn_t).reshape(n_days, n_pairs)
        b_err = np.abs(b_p - b_t).reshape(n_days, n_pairs)

        daily_mae_tgn = tgn_err.mean(axis=1)
        daily_mae_b = b_err.mean(axis=1)

        # DM test with MAE loss
        dm_mae = diebold_mariano_test(
            daily_mae_tgn, daily_mae_b, loss="mae", alternative="less",
        )
        dm_results[f"tgn_vs_{variant_b}_mae"] = dm_mae

        # DM test with MSE loss
        daily_mse_tgn = (tgn_err ** 2).mean(axis=1)
        daily_mse_b = (b_err ** 2).mean(axis=1)
        dm_mse = diebold_mariano_test(
            daily_mse_tgn, daily_mse_b, loss="mae", alternative="less",
        )
        dm_results[f"tgn_vs_{variant_b}_mse"] = dm_mse

        all_p_values[f"dm_mae_tgn_vs_{variant_b}"] = dm_mae["p_value"]
        all_p_values[f"dm_mse_tgn_vs_{variant_b}"] = dm_mse["p_value"]

        logger.info("  DM TGN vs %s (MAE):", name_b)
        logger.info("    N=%d dias, max_lags=%d", dm_mae["n_days"], dm_mae["max_lags_hac"])
        logger.info("    mean(L_TGN - L_%s) = %.6f", name_b, dm_mae["mean_loss_diff"])
        logger.info("    DM stat = %.4f,  p-value = %.4e", dm_mae["dm_statistic"], dm_mae["p_value"])
        logger.info("    Cohen's d = %.4f (%s)", dm_mae["effect_size_d"],
                     _interpret_cohens_d(dm_mae["effect_size_d"]))
        if dm_mae["p_value"] < 0.05:
            logger.info("    >>> TGN PREDITIVAMENTE SUPERIOR (DM-MAE p < 0.05) [PASS]")
        else:
            logger.info("    >>> Diferença NÃO significativa (DM-MAE). [FAIL]")

        logger.info("  DM TGN vs %s (MSE):", name_b)
        logger.info("    DM stat = %.4f,  p-value = %.4e", dm_mse["dm_statistic"], dm_mse["p_value"])
        if dm_mse["p_value"] < 0.05:
            logger.info("    >>> TGN PREDITIVAMENTE SUPERIOR (DM-MSE p < 0.05) [PASS]")
        else:
            logger.info("    >>> Diferença NÃO significativa (DM-MSE). [FAIL]")

    # --- 7. Holm-Bonferroni Correction ---
    logger.info("=" * 60)
    logger.info("CORREÇÃO PARA TESTES MÚLTIPLOS — Holm-Bonferroni")
    logger.info("  Numero total de testes: %d", len(all_p_values))

    holm_results = holm_bonferroni(all_p_values)

    for test_name, info in sorted(holm_results.items(), key=lambda x: x[1]["original_p"]):
        sig = "✓ SIG" if info["significant_at_0.05"] else "✗ n.s."
        logger.info(
            "  [%s] %-40s  p_orig=%.4e  p_corr=%.4e",
            sig, test_name, info["original_p"], info["corrected_p"],
        )

    # Count how many survive correction
    n_sig_original = sum(1 for v in all_p_values.values() if v is not None and v < 0.05)
    n_sig_corrected = sum(1 for v in holm_results.values() if v["significant_at_0.05"])
    logger.info("  Significantes: %d/%d (original) → %d/%d (corrigido Holm)",
                 n_sig_original, len(all_p_values), n_sig_corrected, len(all_p_values))

    # --- 8. Resumo R² ---
    logger.info("=" * 60)
    logger.info("RESUMO R²")
    for v, m in results.items():
        r2 = m.get("r_squared", float("nan"))
        flag = " [>80% TARGET MET]" if v == "tgn" and r2 >= 0.80 else ""
        logger.info("  %-12s R²=%.4f%s", v.upper(), r2, flag)

    # --- 9. Effect Size Summary ---
    logger.info("=" * 60)
    logger.info("RESUMO EFFECT SIZES")
    logger.info("  Interpretação: |d|/|r| < 0.2 = negligível, 0.2-0.5 = pequeno, 0.5-0.8 = médio, > 0.8 = grande")

    if "tgn_vs_roland" in sharpe_diff_results:
        d = sharpe_diff_results["tgn_vs_roland"]["cohens_d"]
        logger.info("  Bootstrap Sharpe (TGN-ROLAND): Cohen's d = %.4f (%s)", d, _interpret_cohens_d(d))

    if "tgn_vs_gat_static" in sharpe_diff_results:
        d = sharpe_diff_results["tgn_vs_gat_static"]["cohens_d"]
        logger.info("  Bootstrap Sharpe (TGN-GAT):    Cohen's d = %.4f (%s)", d, _interpret_cohens_d(d))

    for name, w_res in wilcoxon_results.items():
        if w_res["effect_size_r"] is not None:
            logger.info("  Wilcoxon %s: rank-biserial r = %.4f (%s)",
                         name, w_res["effect_size_r"], _interpret_effect_r(w_res["effect_size_r"]))

    for name, dm_res in dm_results.items():
        logger.info("  DM %s: Cohen's d = %.4f (%s)",
                     name, dm_res["effect_size_d"], _interpret_cohens_d(dm_res["effect_size_d"]))

    # --- 10. Output JSON ---
    out_dir = RESULTS_DIR / f"bootstrap_eval_v3_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "version": "v3.1_robust",
        # Existing metrics (backward compat)
        "metrics": results,
        # Bootstrap individual
        "bootstrap_mean_sharpes": {k: float(np.mean(v)) for k, v in bootstrap_sharpes.items()},
        "bootstrap_ci_2.5": {k: float(np.percentile(v, 2.5)) for k, v in bootstrap_sharpes.items()},
        "bootstrap_ci_97.5": {k: float(np.percentile(v, 97.5)) for k, v in bootstrap_sharpes.items()},
        # Sharpe diff comparisons
        "sharpe_diff_comparisons": sharpe_diff_results,
        # CVaR comparisons
        "cvar_comparisons": cvar_paired_results,
        # Wilcoxon tests
        "wilcoxon_tests": wilcoxon_results,
        # Diebold-Mariano tests
        "diebold_mariano_tests": dm_results,
        # All raw p-values
        "all_p_values": {k: v for k, v in all_p_values.items()},
        # Holm-Bonferroni corrected
        "holm_bonferroni": holm_results,
        # Config
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
        json.dump(summary, f, indent=2, default=str)

    logger.info("Resumo salvo em: %s", out_path)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bootstrap Eval v3 — Robust Statistical Testing")
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
