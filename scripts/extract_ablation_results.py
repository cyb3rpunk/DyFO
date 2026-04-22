"""Extrai resultados de ablação dos link_pred directories e combina com o rev2 summary.

O abllation_test.py só escreve o JSON final quando a fase normal (4 variantes × 9 janelas)
termina. Este script reconstrói o corpo de ablação a partir dos link_pred_tgat directories
já gravados, combinando com o rev2 summary existente (que já tem os resultados de comparação
entre variantes).

Uso:
    python scripts/extract_ablation_results.py
    python scripts/extract_ablation_results.py --rev2_summary results/bootstrap_eval_tkg_rev2_YYYYMMDD_HHMMSS/bootstrap_summary_tkg_rev2.json
    python scripts/extract_ablation_results.py --dry_run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Limite da fase de ablação: diretórios criados ANTES do início da fase normal
NORMAL_PHASE_FIRST_DIR = "link_pred_tgat_s42_20260420_021208"

# Sequência de sets de ablação (modo full) — mesma ordem do ABLATION_FULL em rev2
ABLATION_SEQUENCE: List[Tuple[str, List[str], bool]] = [
    ("CORR_only",  ["CORR"],             False),  # (label, arestas, é_degenerado)
    ("SECT_only",  ["SECT"],             True),
    ("FACT_only",  ["FACT"],             True),
    ("CORR+SECT",  ["CORR", "SECT"],     False),
    ("CORR+FACT",  ["CORR", "FACT"],     False),
    ("SECT+FACT",  ["SECT", "FACT"],     True),
    ("all_edges",  ["CORR", "SECT", "FACT"], False),
]
N_WINDOWS = 9


def _find_ablation_dirs() -> List[Path]:
    """Retorna os link_pred_tgat directories da fase de ablação, ordenados por timestamp."""
    ablation_dirs = sorted([
        d for d in RESULTS_DIR.iterdir()
        if (
            d.name.startswith("link_pred_tgat_s42_20260419") or
            (d.name.startswith("link_pred_tgat_s42_20260420") and d.name < NORMAL_PHASE_FIRST_DIR)
        )
    ])
    return ablation_dirs


def _read_metrics(d: Path) -> Optional[Dict[str, Any]]:
    rj = d / "results.json"
    if not rj.exists():
        return None
    data = json.loads(rj.read_text(encoding="utf-8"))
    return data.get("metrics", {})


def _is_degenerate(metrics: Dict[str, Any]) -> bool:
    return metrics.get("best_epoch", 0) <= 1 and metrics.get("test_r_squared", 0.0) == 0.0


def _assign_dirs_to_sets(dirs: List[Path]) -> Dict[str, List[Optional[Dict[str, Any]]]]:
    """
    Atribui cada diretório ao seu set de ablação e janela usando a ordem cronológica.

    O primeiro diretório (sem results.json) é descartado. Os 63 restantes são atribuídos
    em blocos de 9, respeitando a sequência ABLATION_SEQUENCE.
    """
    valid: List[Tuple[Path, Dict[str, Any]]] = []
    for d in dirs:
        m = _read_metrics(d)
        if m is None:
            print(f"  [SKIP] {d.name} — sem results.json (provável falha na inicialização)")
            continue
        valid.append((d, m))

    total_expected = N_WINDOWS * len(ABLATION_SEQUENCE)
    if len(valid) != total_expected:
        raise RuntimeError(
            f"Esperado {total_expected} runs válidos (7 sets × 9 janelas), "
            f"mas encontrado {len(valid)}. Verifique os diretórios de ablação."
        )

    assigned: Dict[str, List[Optional[Dict[str, Any]]]] = {}
    idx = 0
    for label, edges, is_deg in ABLATION_SEQUENCE:
        windows = []
        for w in range(N_WINDOWS):
            path, metrics = valid[idx]
            actual_deg = _is_degenerate(metrics)
            if actual_deg != is_deg:
                raise RuntimeError(
                    f"Inconsistência no set '{label}' janela {w+1}: "
                    f"esperado {'degenerado' if is_deg else 'válido'}, "
                    f"mas encontrado {'degenerado' if actual_deg else 'válido'} "
                    f"em {path.name}"
                )
            windows.append(metrics)
            idx += 1
        assigned[label] = windows
    return assigned


def _compute_set_stats(label: str, edges: List[str], window_metrics: List[Dict]) -> Dict[str, Any]:
    sharpes = np.array([
        m.get("test_sharpe_proxy", np.nan) for m in window_metrics
    ], dtype=float)
    r2s = np.array([m.get("test_r_squared", 0.0) for m in window_metrics], dtype=float)
    spearmans = np.array([m.get("test_spearman", 0.0) for m in window_metrics], dtype=float)

    def safe_mean(a: np.ndarray) -> Optional[float]:
        v = float(np.nanmean(a))
        return None if np.isnan(v) else v

    def safe_std(a: np.ndarray) -> Optional[float]:
        if np.all(np.isnan(a)):
            return None
        v = float(np.nanstd(a[~np.isnan(a)], ddof=1)) if np.sum(~np.isnan(a)) > 1 else 0.0
        return None if np.isnan(v) else v

    return {
        "active_edges": sorted(edges),
        "mean_sharpe": safe_mean(sharpes),
        "std_sharpe": safe_std(sharpes),
        "mean_r_squared": safe_mean(r2s),
        "mean_spearman": safe_mean(spearmans),
        "window_metrics": [
            {
                "window": i + 1,
                "r_squared": m.get("test_r_squared"),
                "spearman": m.get("test_spearman"),
                "mae": m.get("test_mae"),
                "sharpe_proxy": m.get("test_sharpe_proxy"),
                "cls_f1": m.get("test_cls_f1"),
                "best_epoch": m.get("best_epoch"),
            }
            for i, m in enumerate(window_metrics)
        ],
    }


def _build_ablation_body(assigned: Dict[str, List[Dict]]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for label, edges, _ in ABLATION_SEQUENCE:
        results[label] = _compute_set_stats(label, edges, assigned[label])

    ranking = sorted(
        [(k, v["mean_sharpe"] if v["mean_sharpe"] is not None else float("nan"))
         for k, v in results.items()],
        key=lambda x: (x[1] is None or np.isnan(x[1]), -(x[1] or 0)),
    )

    return {
        "ablation_variant": "tgat",
        "ablation_mode": "full",
        "n_windows": N_WINDOWS,
        "ablation_results": results,
        "ablation_ranking_by_sharpe": ranking,
    }


def _markdown_report(ablation_body: Dict[str, Any], rev2_summary: Dict[str, Any]) -> str:
    cfg = rev2_summary.get("run_config", {})
    desc = rev2_summary.get("descriptive_summary", {})
    mean_sharpe = desc.get("mean_window_sharpe", {})
    ranking = ablation_body["ablation_ranking_by_sharpe"]
    results = ablation_body["ablation_results"]

    lines = [
        "# DyFO — Relatório de Ablação (Extraído)",
        "",
        "> Resultados extraídos dos link_pred directories antes da conclusão do run completo.",
        "> Fase de ablação: CONCLUÍDA. Fase normal: em progresso (TGAT W9 + TGN/Roland/GAT-Static pendentes).",
        "> Para a comparação entre variantes, os dados do run `bootstrap_eval_tkg_rev2_20260418_130703` foram usados.",
        "",
        "## Configuração",
        f"- Tickers: {cfg.get('n_tickers')}",
        f"- Janelas walk-forward: {cfg.get('n_windows')}",
        f"- Ablation variant: tgat (stateless)",
        f"- Ablation mode: full (7 subsets)",
        f"- Período: {cfg.get('start')} → {cfg.get('end')}",
        "",
        "## 1. Ablação de Arestas — Ranking por Sharpe (TGAT)",
        "",
        "| # | Conjunto | Arestas Ativas | Sharpe médio | Std | R² médio | Spearman médio |",
        "|---|----------|---------------|-------------|-----|---------|---------------|",
    ]

    for pos, (label, sharpe_val) in enumerate(ranking, start=1):
        r = results[label]
        sh = f"{r['mean_sharpe']:.4f}" if r["mean_sharpe"] is not None else "nan"
        std = f"{r['std_sharpe']:.4f}" if r["std_sharpe"] is not None else "nan"
        r2 = f"{r['mean_r_squared']:.4f}" if r["mean_r_squared"] is not None else "nan"
        sp = f"{r['mean_spearman']:.4f}" if r["mean_spearman"] is not None else "nan"
        edges_str = " + ".join(r["active_edges"])
        lines.append(f"| {pos} | `{label}` | {edges_str} | **{sh}** | {std} | {r2} | {sp} |")

    lines.extend([
        "",
        "> **Interpretação:** Sets com `nan` (SECT_only, FACT_only, SECT+FACT) produziram grafos",
        "> degenerados sem arestas CORR — o modelo converge para predições triviais em época 1.",
        "> A aresta CORR (DCC-GARCH) é o componente essencial do sinal.",
        "",
        "## 2. Detalhes por Janela — Sets Não-Degenerados",
    ])

    for label, edges, is_deg in ABLATION_SEQUENCE:
        if is_deg:
            continue
        r = results[label]
        lines.extend([
            "",
            f"### {label} ({' + '.join(r['active_edges'])})",
            "",
            "| Janela | R² | Spearman | MAE | Sharpe | F1 | Best epoch |",
            "|--------|-----|---------|-----|--------|-----|-----------|",
        ])
        for wm in r["window_metrics"]:
            r2 = f"{wm['r_squared']:.4f}" if wm['r_squared'] is not None else "—"
            sp = f"{wm['spearman']:.4f}" if wm['spearman'] is not None else "—"
            mae = f"{wm['mae']:.4f}" if wm['mae'] is not None else "—"
            sh = f"{wm['sharpe_proxy']:.4f}" if wm['sharpe_proxy'] is not None else "—"
            f1 = f"{wm['cls_f1']:.4f}" if wm['cls_f1'] is not None else "—"
            ep = wm['best_epoch']
            lines.append(f"| {wm['window']} | {r2} | {sp} | {mae} | {sh} | {f1} | {ep} |")

    lines.extend([
        "",
        "## 3. Comparação Entre Variantes (Rev2, 50 tickers, 9 janelas)",
        "",
        "| Variante | R² médio | Spearman | Sharpe proxy |",
        "|----------|---------|---------|------------|",
    ])
    mean_metrics = desc.get("mean_window_metrics", {})
    for variant in ["tgat", "tgn", "roland", "gat_static"]:
        mm = mean_metrics.get(variant, {})
        r2 = f"{mm.get('r_squared', 0):.4f}"
        sp = f"{mm.get('spearman', 0):.4f}"
        sh = f"{mean_sharpe.get(variant, 0):.4f}"
        lines.append(f"| {variant} | {r2} | {sp} | {sh} |")

    lines.extend([
        "",
        "## 4. Testes Estatísticos (Diebold-Mariano — TGAT vs Baselines)",
        "",
        "| Comparação | DM stat | p-value | Cohen's d | Sig. |",
        "|-----------|---------|---------|---------|-----|",
    ])
    pooled = rev2_summary.get("pooled_predictive_tests", {})
    pairs = [
        ("tgat_vs_tgn_mae",       "TGAT vs TGN"),
        ("tgat_vs_roland_mae",    "TGAT vs Roland"),
        ("tgat_vs_gat_static_mae","TGAT vs GAT-Static"),
    ]
    for key, label in pairs:
        t = pooled.get(key, {})
        dm = f"{t.get('dm_statistic', 0):.2f}"
        pv = t.get("p_value", 1.0)
        pv_str = f"{pv:.2e}" if pv < 0.001 else f"{pv:.4f}"
        d = f"{t.get('effect_size_d', 0):.2f}"
        sig = "✅" if pv < 0.05 else "❌"
        lines.append(f"| {label} | {dm} | {pv_str} | {d} | {sig} |")

    lines.extend([
        "",
        "---",
        f"*Gerado em: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | Script: extract_ablation_results.py*",
    ])
    return "\n".join(lines) + "\n"


def run(rev2_summary_path: Optional[Path] = None, dry_run: bool = False) -> Dict[str, Any]:
    print("=" * 60)
    print("DyFO — Extração de resultados de ablação")
    print("=" * 60)

    # 1. Localizar rev2 summary
    if rev2_summary_path is None:
        candidates = sorted(RESULTS_DIR.glob("bootstrap_eval_tkg_rev2_*/bootstrap_summary_tkg_rev2.json"))
        if not candidates:
            raise FileNotFoundError("Nenhum bootstrap_summary_tkg_rev2.json encontrado em results/")
        rev2_summary_path = candidates[-1]
    print(f"Rev2 summary: {rev2_summary_path}")

    with open(rev2_summary_path, encoding="utf-8") as f:
        rev2_summary = json.load(f)

    # 2. Coletar e mapear diretórios de ablação
    print("\nColetando diretórios de ablação...")
    ablation_dirs = _find_ablation_dirs()
    print(f"  Encontrados: {len(ablation_dirs)} diretórios")

    assigned = _assign_dirs_to_sets(ablation_dirs)
    print("  Mapeamento concluído:")
    for label, _, _ in ABLATION_SEQUENCE:
        n_valid = sum(1 for m in assigned[label] if not _is_degenerate(m))
        print(f"    {label}: {len(assigned[label])} janelas ({n_valid} válidas)")

    # 3. Construir corpo de ablação
    ablation_body = _build_ablation_body(assigned)

    print("\nRanking de ablação (Sharpe):")
    for pos, (label, sharpe_val) in enumerate(ablation_body["ablation_ranking_by_sharpe"], start=1):
        sh_str = f"{sharpe_val:.4f}" if sharpe_val is not None and not np.isnan(sharpe_val) else "nan"
        print(f"  {pos}. {label:<12} mean_sharpe={sh_str}")

    # 4. Combinar com rev2 summary
    combined = {**rev2_summary, "ablation": ablation_body}

    # 5. Escrever outputs
    if dry_run:
        print("\n[DRY RUN] Nenhum arquivo escrito.")
        return combined

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"ablation_extracted_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "ablation_extracted_summary.json"
    report_path = out_dir / "ablation_extracted_report.md"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"\nSummary JSON -> {summary_path}")

    report_md = _markdown_report(ablation_body, rev2_summary)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Markdown report -> {report_path}")

    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrai resultados de ablação dos link_pred dirs")
    parser.add_argument(
        "--rev2_summary",
        type=Path,
        default=None,
        help="Caminho para bootstrap_summary_tkg_rev2.json. Padrão: mais recente em results/",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Apenas exibe o resultado sem escrever arquivos",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    run(rev2_summary_path=args.rev2_summary, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
