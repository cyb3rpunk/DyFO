"""Link prediction decoder and training utilities for self-supervised pre-training.

Two modes supported:
  1) Classification: predict whether |rho(i,j)| at t+1 exceeds threshold (original).
  2) Regression: predict the continuous correlation rho(i,j) at t+1 (preferred).

Regression mode is preferred because it eliminates the threshold sensitivity problem
and provides a richer training signal. See EXPERIMENT_LOG v0.1-v0.3 for motivation.

This follows DyFO Manual §5.2:
    loss_pretrain = L(f(z_i(t), z_j(t)), rho_ij_{t+1})
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss for imbalanced binary classification.

    Down-weights easy examples so the model focuses on hard negatives.
    Lin et al. (2017) - Focal Loss for Dense Object Detection.

    Parameters
    ----------
    logits : Tensor (B,) — raw logits (pre-sigmoid)
    labels : Tensor (B,) — 0 or 1
    alpha : float — weighting factor for positives (1-alpha for negatives)
    gamma : float — focusing parameter (0 = standard BCE)
    """
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * labels + (1 - probs) * (1 - labels)
    alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


class LinkPredictor(nn.Module):
    """MLP decoder for link prediction: p(edge | z_i, z_j).

    Takes concatenated embeddings [z_i || z_j] and outputs a scalar probability.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Predict link probability.

        Parameters
        ----------
        z_i, z_j : Tensor of shape (B, embedding_dim)

        Returns
        -------
        Tensor of shape (B,) — logits (pre-sigmoid).
        """
        h = torch.cat([z_i, z_j], dim=-1)
        return self.net(h).squeeze(-1)

    def predict_proba(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Return probability (sigmoid applied)."""
        return torch.sigmoid(self.forward(z_i, z_j))


def build_link_labels(
    corr_today: dict,
    corr_tomorrow: dict,
    num_nodes: int,
    threshold: float = 0.3,
    neg_ratio: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build positive and negative edge labels for link prediction.

    A positive label means |corr(i,j)| at t+1 exceeds the threshold.
    Negative samples are randomly chosen pairs that do NOT have high correlation.

    Parameters
    ----------
    corr_today : dict
        Mapping (i, j) -> rho for today (used to know which pairs exist).
    corr_tomorrow : dict
        Mapping (i, j) -> rho for tomorrow (labels).
    num_nodes : int
    threshold : float
        Absolute correlation threshold for positive label.
    neg_ratio : float
        Ratio of negative to positive samples.

    Returns
    -------
    src, dst : LongTensor of shape (num_samples,) — node indices
    labels : FloatTensor of shape (num_samples,) — 0 or 1
    """
    positive_src = []
    positive_dst = []

    # Positive edges: high correlation tomorrow
    for (i, j), rho in corr_tomorrow.items():
        if abs(rho) >= threshold:
            positive_src.append(i)
            positive_dst.append(j)

    num_pos = len(positive_src)
    if num_pos == 0:
        return (
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0),
        )

    # Negative edges: random pairs NOT in positive set
    pos_set = set(zip(positive_src, positive_dst))
    neg_src = []
    neg_dst = []
    num_neg = int(num_pos * neg_ratio)

    attempts = 0
    max_attempts = num_neg * 10
    while len(neg_src) < num_neg and attempts < max_attempts:
        i = torch.randint(0, num_nodes, (1,)).item()
        j = torch.randint(0, num_nodes, (1,)).item()
        if i != j and (i, j) not in pos_set and (j, i) not in pos_set:
            neg_src.append(i)
            neg_dst.append(j)
            pos_set.add((i, j))  # avoid duplicates
        attempts += 1

    src = torch.tensor(positive_src + neg_src, dtype=torch.long)
    dst = torch.tensor(positive_dst + neg_dst, dtype=torch.long)
    labels = torch.cat([
        torch.ones(num_pos),
        torch.zeros(len(neg_src)),
    ])

    # Shuffle
    perm = torch.randperm(len(src))
    return src[perm], dst[perm], labels[perm]


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.0,
) -> dict:
    """Compute link prediction metrics: loss, accuracy, AUC-approx, precision, recall.
    
    Parameters
    ----------
    threshold : float
        Logit threshold for classification (0.0 = sigmoid 0.5).
    """
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    preds = (logits > threshold).float()
    acc = (preds == labels).float().mean()

    # Precision / Recall
    tp = ((preds == 1) & (labels == 1)).float().sum()
    fp = ((preds == 1) & (labels == 0)).float().sum()
    fn = ((preds == 0) & (labels == 1)).float().sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Approximate AUC via ranking
    with torch.no_grad():
        try:
            pos_scores = logits[labels == 1]
            neg_scores = logits[labels == 0]
            if len(pos_scores) > 0 and len(neg_scores) > 0:
                # Count how many pos scores > neg scores
                comparisons = (pos_scores.unsqueeze(1) > neg_scores.unsqueeze(0)).float()
                auc = comparisons.mean()
            else:
                auc = torch.tensor(0.5)
        except Exception:
            auc = torch.tensor(0.5)

    return {
        "loss": loss.item(),
        "accuracy": acc.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "auc": auc.item(),
    }


# =========================================================================
# Regression mode — predict continuous rho(i,j) at t+1
# =========================================================================


class CorrelationRegressor(nn.Module):
    """MLP decoder for correlation regression: rho_hat = f(z_i, z_j).

    Takes concatenated embeddings [z_i || z_j] and outputs a scalar in [-1, 1].
    Uses tanh activation on the output to bound predictions.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Predict correlation value.

        Parameters
        ----------
        z_i, z_j : Tensor of shape (B, embedding_dim)

        Returns
        -------
        Tensor of shape (B,) — predicted rho in [-1, 1] via tanh.
        """
        h = torch.cat([z_i, z_j], dim=-1)
        return torch.tanh(self.net(h).squeeze(-1))


def build_regression_labels(
    corr_tomorrow: dict,
    num_nodes: int,
    sample_ratio: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build regression labels: predict continuous rho for all known pairs.

    For pairs without correlation data tomorrow, we skip them (unknown, not zero).

    Parameters
    ----------
    corr_tomorrow : dict
        Mapping (i, j) -> rho for tomorrow.
    num_nodes : int
    sample_ratio : float
        Fraction of available pairs to use (1.0 = all). For large graphs,
        reduce to control computation.

    Returns
    -------
    src, dst : LongTensor of shape (num_samples,)
    rho_values : FloatTensor of shape (num_samples,) — continuous in [-1, 1]
    """
    seen = set()
    src_list = []
    dst_list = []
    rho_list = []

    for (i, j), rho in corr_tomorrow.items():
        pair = (min(i, j), max(i, j))
        if pair in seen:
            continue
        seen.add(pair)
        src_list.append(pair[0])
        dst_list.append(pair[1])
        rho_list.append(rho)

    if not src_list:
        return (
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0),
        )

    src = torch.tensor(src_list, dtype=torch.long)
    dst = torch.tensor(dst_list, dtype=torch.long)
    rho_values = torch.tensor(rho_list, dtype=torch.float)

    # Optional subsampling
    if sample_ratio < 1.0:
        n = max(1, int(len(src) * sample_ratio))
        perm = torch.randperm(len(src))[:n]
        src, dst, rho_values = src[perm], dst[perm], rho_values[perm]

    # NB: Não aplicar shuffle — a ordem determinística (sorted unique pairs)
    # é necessária para alinhamento cross-model no Wilcoxon signed-rank test.
    # O Huber loss é indiferente à ordem dos samples.
    return src, dst, rho_values


def compute_regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    classification_threshold: float = 0.5,
) -> dict:
    """Compute regression metrics for correlation prediction.

    Primary: MSE, MAE, R-squared, rank correlation (Spearman approx).
    Secondary: classification metrics at given threshold (for backward compatibility).

    Parameters
    ----------
    predictions : Tensor (B,) — predicted rho in [-1, 1]
    targets : Tensor (B,) — actual rho
    classification_threshold : float
        |rho| threshold for derived classification metrics.
    """
    mse = F.mse_loss(predictions, targets)
    mae = F.l1_loss(predictions, targets)

    # R-squared
    ss_res = ((targets - predictions) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r_squared = 1.0 - ss_res / (ss_tot + 1e-8)

    # Rank correlation (Spearman approximation via Pearson on ranks)
    with torch.no_grad():
        n = len(predictions)
        if n > 2:
            pred_ranks = predictions.argsort().argsort().float()
            target_ranks = targets.argsort().argsort().float()
            rank_corr_num = ((pred_ranks - pred_ranks.mean()) * (target_ranks - target_ranks.mean())).sum()
            rank_corr_den = (
                ((pred_ranks - pred_ranks.mean()) ** 2).sum().sqrt()
                * ((target_ranks - target_ranks.mean()) ** 2).sum().sqrt()
            )
            spearman = rank_corr_num / (rank_corr_den + 1e-8)
        else:
            spearman = torch.tensor(0.0)

    # Derived classification metrics (for comparison with previous experiments)
    with torch.no_grad():
        pred_class = (predictions.abs() >= classification_threshold).float()
        target_class = (targets.abs() >= classification_threshold).float()
        tp = ((pred_class == 1) & (target_class == 1)).float().sum()
        fp = ((pred_class == 1) & (target_class == 0)).float().sum()
        fn = ((pred_class == 0) & (target_class == 1)).float().sum()
        cls_precision = tp / (tp + fp + 1e-8)
        cls_recall = tp / (tp + fn + 1e-8)
        cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall + 1e-8)
        cls_accuracy = (pred_class == target_class).float().mean()

    return {
        "loss": mse.item(),
        "mse": mse.item(),
        "mae": mae.item(),
        "r_squared": r_squared.item(),
        "spearman": spearman.item(),
        "cls_accuracy": cls_accuracy.item(),
        "cls_precision": cls_precision.item(),
        "cls_recall": cls_recall.item(),
        "cls_f1": cls_f1.item(),
    }
