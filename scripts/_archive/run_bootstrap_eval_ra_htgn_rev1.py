"""Recommended BL-17 follow-up runner for RA-HTGN.

This revision keeps the same evaluation flow as ``run_bootstrap_eval_ra_htgn.py``
but bakes in a higher-power default configuration for the next experiment:

- more training epochs
- denser walk-forward stepping to target more windows
- more bootstrap iterations
- configurable learning rates and patience from the CLI
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.run_bootstrap_eval_ra_htgn as base_runner


DEFAULT_START = "2020-01-01"
DEFAULT_END = "2024-12-31"
DEFAULT_EPOCHS = 50
DEFAULT_TRAIN_DAYS = 500
DEFAULT_VAL_DAYS = 125
DEFAULT_TEST_DAYS = 125
DEFAULT_STEP_DAYS = 125
DEFAULT_N_BOOTSTRAP = 2000
DEFAULT_BLOCK_SIZE = 5
DEFAULT_MAX_WINDOWS = 10
DEFAULT_TGN_LR = 1e-3
DEFAULT_BASELINE_LR = 1e-3
DEFAULT_TGN_PATIENCE = 8
DEFAULT_BASELINE_PATIENCE = 8


def run_bootstrap_eval_ra_htgn_rev1(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    epochs: int = DEFAULT_EPOCHS,
    train_days: int = DEFAULT_TRAIN_DAYS,
    val_days: int = DEFAULT_VAL_DAYS,
    test_days: int = DEFAULT_TEST_DAYS,
    step_days: int = DEFAULT_STEP_DAYS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    block_size: int = DEFAULT_BLOCK_SIZE,
    max_windows: int | None = DEFAULT_MAX_WINDOWS,
    tgn_lr: float = DEFAULT_TGN_LR,
    baseline_lr: float = DEFAULT_BASELINE_LR,
    tgn_patience: int = DEFAULT_TGN_PATIENCE,
    baseline_patience: int = DEFAULT_BASELINE_PATIENCE,
):
    # Patch the shared runner constants so the existing training flow uses the
    # recommended hyperparameters without duplicating the whole evaluation code.
    base_runner.TGN_LR = tgn_lr
    base_runner.BASELINE_LR = baseline_lr
    base_runner.TGN_PATIENCE = tgn_patience
    base_runner.BASELINE_PATIENCE = baseline_patience

    return base_runner.run_bootstrap_eval_ra_htgn(
        start=start,
        end=end,
        epochs=epochs,
        train_days=train_days,
        val_days=val_days,
        test_days=test_days,
        step_days=step_days,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        max_windows=max_windows,
    )


def main():
    parser = argparse.ArgumentParser(
        description="DyFO bootstrap eval for BL-17 RA-HTGN (rev1 recommended config)"
    )
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--train_days", type=int, default=DEFAULT_TRAIN_DAYS)
    parser.add_argument("--val_days", type=int, default=DEFAULT_VAL_DAYS)
    parser.add_argument("--test_days", type=int, default=DEFAULT_TEST_DAYS)
    parser.add_argument("--step_days", type=int, default=DEFAULT_STEP_DAYS)
    parser.add_argument("--n_bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--block_size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--max_windows", type=int, default=DEFAULT_MAX_WINDOWS)
    parser.add_argument("--tgn_lr", type=float, default=DEFAULT_TGN_LR)
    parser.add_argument("--baseline_lr", type=float, default=DEFAULT_BASELINE_LR)
    parser.add_argument("--tgn_patience", type=int, default=DEFAULT_TGN_PATIENCE)
    parser.add_argument("--baseline_patience", type=int, default=DEFAULT_BASELINE_PATIENCE)
    args = parser.parse_args()

    run_bootstrap_eval_ra_htgn_rev1(
        start=args.start,
        end=args.end,
        epochs=args.epochs,
        train_days=args.train_days,
        val_days=args.val_days,
        test_days=args.test_days,
        step_days=args.step_days,
        n_bootstrap=args.n_bootstrap,
        block_size=args.block_size,
        max_windows=args.max_windows,
        tgn_lr=args.tgn_lr,
        baseline_lr=args.baseline_lr,
        tgn_patience=args.tgn_patience,
        baseline_patience=args.baseline_patience,
    )


if __name__ == "__main__":
    main()
