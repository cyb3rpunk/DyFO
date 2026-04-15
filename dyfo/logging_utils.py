"""Logging and result persistence for DyFO experiments."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = _PROJECT_ROOT / "logs"
RESULTS_DIR = _PROJECT_ROOT / "results"


def _ensure_dirs():
    LOGS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

def setup_logging(
    name: str = "dyfo",
    level: int = logging.INFO,
    log_to_file: bool = True,
    run_tag: Optional[str] = None,
) -> logging.Logger:
    """Configure structured logging to console and file.

    Parameters
    ----------
    name : str
        Logger name.
    level : int
        Logging level.
    log_to_file : bool
        If True, also writes to logs/<run_tag>.log.
    run_tag : str, optional
        Identifier for this run. Defaults to timestamp.

    Returns
    -------
    logging.Logger
    """
    _ensure_dirs()

    if run_tag is None:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler
    if log_to_file:
        log_path = LOGS_DIR / f"{run_tag}.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.info("Log file: %s", log_path)

    return logger


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

class ResultLogger:
    """Saves experiment results as JSON + tensor artifacts.

    Usage:
        rl = ResultLogger("my_experiment")
        rl.log_params({"tickers": [...], "start": "2020-01-01"})
        rl.log_metric("num_events", 12345)
        rl.log_tensor("e_t_final", e_t)
        rl.save()
    """

    def __init__(self, run_tag: Optional[str] = None):
        _ensure_dirs()
        self.run_tag = run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = RESULTS_DIR / self.run_tag
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._data: Dict[str, Any] = {
            "run_tag": self.run_tag,
            "timestamp": datetime.now().isoformat(),
            "params": {},
            "metrics": {},
            "artifacts": [],
        }

    def log_params(self, params: Dict[str, Any]):
        """Log experiment parameters (config, tickers, dates, etc.)."""
        # Ensure JSON-serializable
        for k, v in params.items():
            if isinstance(v, Path):
                params[k] = str(v)
        self._data["params"].update(params)

    def log_metric(self, key: str, value: Any):
        """Log a single scalar metric."""
        if isinstance(value, torch.Tensor):
            value = value.item()
        self._data["metrics"][key] = value

    def log_metrics(self, metrics: Dict[str, Any]):
        """Log multiple metrics at once."""
        for k, v in metrics.items():
            self.log_metric(k, v)

    def log_tensor(self, name: str, tensor: torch.Tensor):
        """Save a tensor artifact to disk."""
        path = self.run_dir / f"{name}.pt"
        torch.save(tensor, path)
        self._data["artifacts"].append(
            {"name": name, "path": str(path), "shape": list(tensor.shape)}
        )

    def log_dataframe(self, name: str, df):
        """Save a pandas DataFrame as CSV."""
        path = self.run_dir / f"{name}.csv"
        df.to_csv(path)
        self._data["artifacts"].append(
            {"name": name, "path": str(path), "shape": list(df.shape)}
        )

    def save(self):
        """Write the result summary JSON."""
        path = self.run_dir / "results.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, default=str)
        return path

    def __repr__(self):
        n_metrics = len(self._data["metrics"])
        n_artifacts = len(self._data["artifacts"])
        return f"ResultLogger('{self.run_tag}', metrics={n_metrics}, artifacts={n_artifacts})"
