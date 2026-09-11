"""Проверка модели на невиданных данных.

Здесь два способа проверки, и они дают разные ответы. Это не дублирование, а
главный урок проекта: случайная нарезка завышает качество, потому что позволяет
учиться на будущем и предсказывать прошлое. Разбиение по времени повторяет то,
что происходит в проде, и число получается честным, но скучным.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import CVConfig
from .metrics import gini_stability, score_panel, stability_components
from .model import train_fold


@dataclass
class CVResult:
    """Всё, что осталось после прогона проверки."""

    oof_predictions: np.ndarray
    evaluated: np.ndarray  # какие строки вообще получили предсказание
    fold_scores: list[float]
    feature_importance: pd.DataFrame
    panel: dict[str, float] = field(default_factory=dict)
    stability: float | None = None
    components: dict | None = None  # из чего сложилась устойчивость

    @property
    def mean_auc(self) -> float:
        return float(np.mean(self.fold_scores))

    @property
    def std_auc(self) -> float:
        """Разброс между фолдами. Большой разброс означает, что среднему верить нельзя."""
        return float(np.std(self.fold_scores))


def time_splits(
    weeks: np.ndarray,
    n_splits: int = 4,
    min_train_share: float = 0.5,
    window: str = "expanding",
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Разбиение по времени. Ни одна строка из будущего не попадает в обучение.

    Режим ``expanding``: обучающая часть растёт от фолда к фолду. Ближе к жизни,
    где данных со временем становится больше, но плохо годится для измерения
    деградации: поздние фолды выигрывают за счёт объёма, а не за счёт времени.

    Режим ``sliding``: окно обучения одной и той же ширины едет вперёд вместе с
    проверочным. Объём обучения постоянен, поэтому разница между фолдами это
    только сдвиг во времени. Именно так проверяют дрейф.
    """
    if window not in ("expanding", "sliding"):
        raise ValueError(f"неизвестный режим окна: {window!r}")

    unique_weeks = np.unique(weeks)
    start = int(len(unique_weeks) * min_train_share)
    edges = np.linspace(start, len(unique_weeks), n_splits + 1).astype(int)

    for i in range(n_splits):
        low, high = edges[i], edges[i + 1]
        if high <= low:
            continue

        train_weeks = unique_weeks[:low] if window == "expanding" else unique_weeks[low - start : low]

        train_mask = np.isin(weeks, train_weeks)
        valid_mask = np.isin(weeks, unique_weeks[low:high])
        if valid_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        yield np.flatnonzero(train_mask), np.flatnonzero(valid_mask)


# Прежнее имя, чтобы не ломать существующие вызовы.
def expanding_window_splits(weeks, n_splits: int = 4, min_train_share: float = 0.5):
    return time_splits(weeks, n_splits, min_train_share, window="expanding")


def _run(
    x: pd.DataFrame,
    y: pd.Series,
    weeks: np.ndarray | None,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> CVResult:
    """Общий цикл: обучить модель на каждом разбиении и собрать предсказания."""
    oof = np.full(len(x), np.nan)
    fold_scores: list[float] = []
    importances = []

    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        booster = train_fold(
            x.iloc[train_idx], y.iloc[train_idx], x.iloc[valid_idx], y.iloc[valid_idx]
        )

        oof[valid_idx] = booster.predict(x.iloc[valid_idx])
        fold_scores.append(score_panel(y.iloc[valid_idx], oof[valid_idx])["roc_auc"])

        importances.append(
            pd.DataFrame(
                {
                    "feature": booster.feature_name(),
                    "gain": booster.feature_importance("gain"),
                    "fold": fold,
                }
            )
        )

    importance = (
        pd.concat(importances)
        .groupby("feature", as_index=False)["gain"]
        .mean()
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )

    evaluated = ~np.isnan(oof)
    result = CVResult(
        oof_predictions=oof,
        evaluated=evaluated,
        fold_scores=fold_scores,
        feature_importance=importance,
        panel=score_panel(y[evaluated], oof[evaluated]),
    )

    if weeks is not None:
        result.components = stability_components(
            weeks[evaluated], y[evaluated].to_numpy(), oof[evaluated]
        )
        result.stability = result.components["stability"]

    return result


def run_time_cv(
    x: pd.DataFrame,
    y: pd.Series,
    weeks: np.ndarray,
    n_splits: int = 4,
    min_train_share: float = 0.5,
    window: str = "expanding",
) -> CVResult:
    """Честная проверка: учимся на прошлом, проверяемся на будущем."""
    splits = list(time_splits(weeks, n_splits, min_train_share, window))
    if not splits:
        raise ValueError("не удалось построить ни одного разбиения по времени")
    return _run(x, y, weeks, splits)


def run_random_cv(
    x: pd.DataFrame, y: pd.Series, weeks: np.ndarray | None = None, config: CVConfig | None = None
) -> CVResult:
    """Случайная стратифицированная нарезка.

    Нужна здесь только для сравнения. Она перемешивает недели, то есть даёт
    модели подглядеть в будущее, и потому систематически завышает качество.
    """
    config = config or CVConfig()
    splitter = StratifiedKFold(
        n_splits=config.n_splits, shuffle=config.shuffle, random_state=config.seed
    )
    return _run(x, y, weeks, list(splitter.split(x, y)))


def holdout_split(weeks: np.ndarray, holdout_from: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Одна отсечка по времени: всё до недели N в обучение, всё с неё в проверку.

    В отличие от фолдов, проверочная часть здесь одна и она большая. Это ближе
    всего к тому, как модель живёт в проде: обучили один раз на накопленном
    прошлом и дальше месяцами применяем к новым заявкам, не переобучая.
    """
    train_mask = weeks < holdout_from
    valid_mask = weeks >= holdout_from

    if train_mask.sum() == 0 or valid_mask.sum() == 0:
        raise ValueError(f"отсечка на неделе {holdout_from} оставляет одну из частей пустой")

    return [(np.flatnonzero(train_mask), np.flatnonzero(valid_mask))]


def run_holdout(
    x: pd.DataFrame, y: pd.Series, weeks: np.ndarray, holdout_from: int
) -> CVResult:
    """Обучить одну модель на прошлом и проверить её на всём будущем сразу."""
    return _run(x, y, weeks, holdout_split(weeks, holdout_from))
