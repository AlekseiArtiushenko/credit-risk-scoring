"""Scoring functions for a binary credit-default model.

ROC AUC is the competition metric, but on its own it hides things a risk team
cares about, so we report a small panel instead of a single number.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Gini coefficient, the form banks usually quote instead of AUC."""
    return 2.0 * roc_auc_score(y_true, y_score) - 1.0


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov gap: the widest separation of good and bad CDFs."""
    order = np.argsort(y_score)
    y_sorted = np.asarray(y_true)[order]

    bad = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    good = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    return float(np.max(np.abs(good - bad)))


def default_rate_in_top_decile(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Observed default rate among the riskiest 10 percent of applicants.

    This is the number that decides whether a cut-off policy is worth running.
    """
    y_true = np.asarray(y_true)
    cutoff = int(len(y_score) * 0.1)
    if cutoff == 0:
        return float("nan")

    riskiest = np.argsort(y_score)[::-1][:cutoff]
    return float(y_true[riskiest].mean())


def score_panel(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """All headline metrics in one dict, ready to print or log."""
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "gini": gini(y_true, y_score),
        "ks": ks_statistic(y_true, y_score),
        "base_rate": float(np.mean(y_true)),
        "top_decile_default_rate": default_rate_in_top_decile(y_true, y_score),
    }
