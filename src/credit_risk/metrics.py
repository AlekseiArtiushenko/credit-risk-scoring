"""Scoring functions for a binary credit-default model.

The competition metric is gini stability: ranking quality that must hold up
week after week rather than peak once. The rest of the panel exists to explain
a stability score, because a single number never says why it moved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
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


def gini_per_week(
    week: np.ndarray, y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Gini computed separately inside each week.

    Weeks where every applicant has the same label carry no ranking
    information, so they are dropped rather than scored as zero.
    """
    frame = pd.DataFrame({"week": week, "y": y_true, "score": y_score})

    weeks: list[int] = []
    ginis: list[float] = []
    for value, group in frame.sort_values("week").groupby("week", sort=True):
        if group["y"].nunique() < 2:
            continue
        weeks.append(value)
        ginis.append(2.0 * roc_auc_score(group["y"], group["score"]) - 1.0)

    return np.asarray(weeks), np.asarray(ginis)


def gini_stability(
    week: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    falling_rate_weight: float = 88.0,
    residual_std_weight: float = 0.5,
) -> float:
    """The competition metric: accuracy that has to survive over time.

    A straight line is fitted through the weekly Gini scores. A downward slope
    is punished hard, and scatter around the line is punished mildly. A model
    that is brilliant today and mediocre in three months scores worse than a
    model that is merely decent every week.

        stability = mean(gini) + 88.0 * min(0, slope) - 0.5 * std(residuals)
    """
    _, ginis = gini_per_week(week, y_true, y_score)
    if len(ginis) < 2:
        raise ValueError("need at least two scorable weeks to measure stability")

    x = np.arange(len(ginis))
    slope, intercept = np.polyfit(x, ginis, 1)
    residuals = ginis - (slope * x + intercept)

    return float(
        np.mean(ginis)
        + falling_rate_weight * min(0.0, slope)
        - residual_std_weight * np.std(residuals)
    )
