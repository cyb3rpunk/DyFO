#!/usr/bin/env python3
"""Compatible COVID forecast experiment for TGAT and autoregressive baselines.

This script evaluates all models on the same prediction table:

  date, src, dst, pred, true

The TGAT prediction comes from ``train_link_prediction.py --save_preds_path``.
EWMA and Persistence are rebuilt causally from the historical target series for
each pair. DCC-GARCH is reported as the label-source/oracle row, because in the
current DyFO pipeline the regression target itself is a DCC-GARCH correlation.
That row is useful as a sanity upper bound, but should not be interpreted as a
fair forecasting baseline against TGAT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - float(np.sum((y - pred) ** 2)) / ss_tot


def _metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    err = pred - y
    sp = spearmanr(y, pred).correlation
    return {
        "n": int(len(y)),
        "r_squared": _r2(y, pred),
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err ** 2)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "spearman": float(sp) if sp is not None else float("nan"),
    }


def _add_causal_baselines(df: pd.DataFrame, ewma_alpha: float) -> pd.DataFrame:
    out = df.sort_values(["pair", "date"]).copy()
    out["pred_persistence"] = out.groupby("pair")["true"].shift(1)

    def ewma_prev(s: pd.Series) -> pd.Series:
        # Prediction for date t uses only labels observed before t.
        return s.shift(1).ewm(alpha=ewma_alpha, adjust=False).mean()

    out["pred_ewma"] = out.groupby("pair")["true"].transform(ewma_prev)
    out["pred_dcc_garch_oracle"] = out["true"]
    return out


def _format_metric(v: float) -> str:
    if pd.isna(v):
        return "nan"
    return f"{v:.6f}"


def _markdown_table(summary: pd.DataFrame) -> str:
    cols = ["model", "role", "n", "r_squared", "mae", "rmse", "spearman"]
    lines = ["| Model | Role | N | R2 | MAE | RMSE | Spearman |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for _, row in summary[cols].iterrows():
        lines.append(
            "| "
            f"{row['model']} | {row['role']} | {int(row['n'])} | "
            f"{_format_metric(row['r_squared'])} | "
            f"{_format_metric(row['mae'])} | "
            f"{_format_metric(row['rmse'])} | "
            f"{_format_metric(row['spearman'])} |"
        )
    return "\n".join(lines)


def run_experiment(
    tgat_preds_path: Path,
    out_dir: Path,
    ewma_alpha: float,
) -> None:
    df = pd.read_csv(tgat_preds_path, parse_dates=["date"])
    required = {"date", "src", "dst", "pred", "true"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {tgat_preds_path}: {sorted(missing)}")

    a = df["src"].astype(str)
    b = df["dst"].astype(str)
    df["pair"] = np.where(a <= b, a + "_" + b, b + "_" + a)
    df = df[["date", "pair", "src", "dst", "pred", "true"]].rename(
        columns={"pred": "pred_tgat"}
    )
    df = _add_causal_baselines(df, ewma_alpha=ewma_alpha)

    model_specs = [
        ("TGAT", "trained forecast", "pred_tgat"),
        ("EWMA", f"causal baseline, alpha={ewma_alpha}", "pred_ewma"),
        ("Persistence", "causal baseline, rho(t-1)", "pred_persistence"),
        ("DCC-GARCH", "label source / oracle upper bound", "pred_dcc_garch_oracle"),
    ]

    # Common sample keeps the model comparison on identical date-pair rows.
    needed_cols = ["true"] + [col for _, _, col in model_specs]
    common = df.dropna(subset=needed_cols).copy()

    rows = []
    for model, role, col in model_specs:
        m = _metrics(common["true"].to_numpy(float), common[col].to_numpy(float))
        rows.append({"model": model, "role": role, **m})

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        by=["role", "r_squared"],
        ascending=[True, False],
        kind="stable",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "covid_compatible_summary.csv"
    rows_path = out_dir / "covid_compatible_predictions.csv"
    report_path = out_dir / "REPORT.md"
    meta_path = out_dir / "metadata.json"

    summary.to_csv(summary_path, index=False)
    common.to_csv(rows_path, index=False)

    metadata = {
        "input_predictions": str(tgat_preds_path),
        "ewma_alpha": ewma_alpha,
        "rows_input": int(len(df)),
        "rows_common": int(len(common)),
        "n_pairs": int(common["pair"].nunique()),
        "n_dates": int(common["date"].nunique()),
        "date_min": common["date"].min().date().isoformat(),
        "date_max": common["date"].max().date().isoformat(),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report = f"""# COVID Compatible Forecast Experiment

Input TGAT predictions: `{tgat_preds_path}`

This experiment puts `TGAT`, `EWMA`, `Persistence`, and `DCC-GARCH` on the same
date-pair panel and the same target: the exported `true` correlation from
`train_link_prediction.py`.

## Sample

- Common rows: `{metadata["rows_common"]}`
- Unique pairs: `{metadata["n_pairs"]}`
- Unique dates: `{metadata["n_dates"]}`
- Date range: `{metadata["date_min"]}` to `{metadata["date_max"]}`
- EWMA alpha: `{ewma_alpha}`

## Metrics

{_markdown_table(summary)}

## Interpretation

`TGAT` is the trained model forecast exported by `train_link_prediction.py`.
`EWMA` and `Persistence` are reconstructed causally from prior target values
for each pair. The `DCC-GARCH` row is not a fair competing forecast here: the
target itself is a DCC-GARCH correlation, so this row is an oracle/label-source
upper bound and mainly verifies that the metric implementation is aligned.

For a paper-style comparison, focus on `TGAT` vs `EWMA` vs `Persistence`. Use
the `DCC-GARCH` row to explain what the labels represent, not as evidence that
DCC-GARCH outperforms the learned model.

## Artifacts

- Summary table: `{summary_path}`
- Common prediction panel: `{rows_path}`
- Metadata: `{meta_path}`
"""
    report_path.write_text(report, encoding="utf-8")

    print(f"Saved summary -> {summary_path}")
    print(f"Saved common predictions -> {rows_path}")
    print(f"Saved report -> {report_path}")
    print(_markdown_table(summary))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compatible COVID comparison: TGAT vs EWMA vs Persistence vs DCC-GARCH label source",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tgat_preds", default=str(ROOT / "results" / "covid_tgat_preds.csv"))
    parser.add_argument("--ewma_alpha", type=float, default=0.05)
    parser.add_argument(
        "--out_dir",
        default=str(ROOT / "results" / "covid_compatible_forecast_experiment"),
    )
    args = parser.parse_args()

    run_experiment(
        tgat_preds_path=Path(args.tgat_preds),
        out_dir=Path(args.out_dir),
        ewma_alpha=args.ewma_alpha,
    )


if __name__ == "__main__":
    main()
