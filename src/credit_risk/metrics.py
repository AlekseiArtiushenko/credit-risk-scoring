"""Метрики для модели дефолта.

Метрика соревнования это gini stability: качество ранжирования, которое должно
держаться неделя за неделей, а не выстрелить один раз. Остальные метрики нужны,
чтобы объяснить, почему оценка сдвинулась, одно число этого никогда не говорит.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Коэффициент Джини. В банках обычно называют качество именно им."""
    return 2.0 * roc_auc_score(y_true, y_score) - 1.0


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Статистика Колмогорова-Смирнова: максимальный разрыв между распределениями
    хороших и плохих заёмщиков по скору."""
    order = np.argsort(y_score)
    y_sorted = np.asarray(y_true)[order]

    bad = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    good = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    return float(np.max(np.abs(good - bad)))


def default_rate_in_top_decile(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Доля дефолтов среди десяти процентов самых рискованных заявок.

    Это число, по которому решают, стоит ли вообще вводить отсечку по скору.
    """
    y_true = np.asarray(y_true)
    cutoff = int(len(y_score) * 0.1)
    if cutoff == 0:
        return float("nan")

    riskiest = np.argsort(y_score)[::-1][:cutoff]
    return float(y_true[riskiest].mean())


def score_panel(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Все основные метрики одним словарём, готовым к печати или записи."""
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
    """Джини, посчитанный отдельно внутри каждой недели.

    Недели, где у всех заявок одинаковая метка, не несут информации о
    ранжировании, поэтому выбрасываются, а не засчитываются как ноль.
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
    """Метрика соревнования: качество, которое обязано пережить время.

    Через недельные значения джини проводится прямая. Отрицательный наклон
    штрафуется жёстко, разброс вокруг прямой штрафуется мягко. Модель, которая
    блестит сегодня и посредственна через три месяца, стоит меньше, чем ровная
    и более слабая.

        stability = mean(gini) + 88.0 * min(0, наклон) - 0.5 * std(остатков)
    """
    _, ginis = gini_per_week(week, y_true, y_score)
    if len(ginis) < 2:
        raise ValueError("нужно минимум две недели с обеими метками")

    x = np.arange(len(ginis))
    slope, intercept = np.polyfit(x, ginis, 1)
    residuals = ginis - (slope * x + intercept)

    return float(
        np.mean(ginis)
        + falling_rate_weight * min(0.0, slope)
        - residual_std_weight * np.std(residuals)
    )
