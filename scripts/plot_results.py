"""Visualise link prediction training results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def load_results(run_dir: Path) -> dict:
    with open(run_dir / "results.json") as f:
        results = json.load(f)
    with open(run_dir / "history.json") as f:
        history = json.load(f)
    return results, history


def plot_training(run_dir: Path, save_path: Path | None = None):
    results, history = load_results(run_dir)

    train = history["train"]
    val = history["val"]
    epochs = list(range(1, len(train) + 1))

    fig = plt.figure(figsize=(16, 12), facecolor="white")
    fig.suptitle(
        "DyFO — Link Prediction Pre-training Results",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3, top=0.92, bottom=0.08)

    # --- 1. Loss ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, [m["loss"] for m in train], "o-", color="#2196F3", label="Train", linewidth=2)
    ax1.plot(epochs, [m["loss"] for m in val], "s-", color="#FF5722", label="Val", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs)

    # --- 2. AUC ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, [m["auc"] for m in train], "o-", color="#2196F3", label="Train", linewidth=2)
    ax2.plot(epochs, [m["auc"] for m in val], "s-", color="#FF5722", label="Val", linewidth=2)
    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    test_auc = results["metrics"].get("test_auc", 0)
    ax2.axhline(y=test_auc, color="#4CAF50", linestyle=":", linewidth=2, label=f"Test={test_auc:.3f}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC")
    ax2.set_title("AUC (Area Under Curve)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs)
    ax2.set_ylim(0.4, 1.0)

    # --- 3. F1 Score ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs, [m["f1"] for m in train], "o-", color="#2196F3", label="Train", linewidth=2)
    ax3.plot(epochs, [m["f1"] for m in val], "s-", color="#FF5722", label="Val", linewidth=2)
    test_f1 = results["metrics"].get("test_f1", 0)
    ax3.axhline(y=test_f1, color="#4CAF50", linestyle=":", linewidth=2, label=f"Test={test_f1:.3f}")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("F1")
    ax3.set_title("F1 Score")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(epochs)

    # --- 4. Precision & Recall ---
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, [m["precision"] for m in train], "o-", color="#9C27B0", label="Train Prec", linewidth=2)
    ax4.plot(epochs, [m["recall"] for m in train], "^-", color="#009688", label="Train Recall", linewidth=2)
    ax4.plot(epochs, [m["precision"] for m in val], "s--", color="#9C27B0", alpha=0.6, label="Val Prec")
    ax4.plot(epochs, [m["recall"] for m in val], "v--", color="#009688", alpha=0.6, label="Val Recall")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Score")
    ax4.set_title("Precision & Recall")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(epochs)

    # --- 5. Accuracy ---
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(epochs, [m["accuracy"] for m in train], "o-", color="#2196F3", label="Train", linewidth=2)
    ax5.plot(epochs, [m["accuracy"] for m in val], "s-", color="#FF5722", label="Val", linewidth=2)
    test_acc = results["metrics"].get("test_accuracy", 0)
    ax5.axhline(y=test_acc, color="#4CAF50", linestyle=":", linewidth=2, label=f"Test={test_acc:.3f}")
    ax5.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Accuracy")
    ax5.set_title("Accuracy")
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_xticks(epochs)

    # --- 6. Summary table ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    params = results.get("params", {})
    metrics = results.get("metrics", {})

    summary_text = (
        f"Configuration\n"
        f"{'─' * 30}\n"
        f"Tickers:     {len(params.get('tickers', []))}\n"
        f"Period:      {params.get('start', '?')} → {params.get('end', '?')}\n"
        f"Parameters:  {metrics.get('total_params', '?'):,}\n"
        f"Epochs:      {params.get('num_epochs', '?')}\n"
        f"LR:          {params.get('lr', '?')}\n"
        f"\n"
        f"Test Results\n"
        f"{'─' * 30}\n"
        f"AUC:         {metrics.get('test_auc', 0):.4f}\n"
        f"F1:          {metrics.get('test_f1', 0):.4f}\n"
        f"Precision:   {metrics.get('test_precision', 0):.4f}\n"
        f"Recall:      {metrics.get('test_recall', 0):.4f}\n"
        f"Accuracy:    {metrics.get('test_accuracy', 0):.4f}\n"
        f"Best epoch:  {metrics.get('best_epoch', '?')}\n"
    )
    ax6.text(
        0.05, 0.95, summary_text,
        transform=ax6.transAxes,
        fontsize=11,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#cccccc"),
    )

    if save_path is None:
        save_path = run_dir / "training_results.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {save_path}")
    return save_path


if __name__ == "__main__":
    import sys

    # Find the most recent link_pred run
    results_dir = Path(__file__).resolve().parent.parent / "results"
    runs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("link_pred")])

    if not runs:
        print("No link prediction results found in results/")
        sys.exit(1)

    run_dir = runs[-1]
    print(f"Plotting results from: {run_dir.name}")
    plot_training(run_dir)
