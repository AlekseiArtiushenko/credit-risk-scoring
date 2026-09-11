"""Проверки метрики устойчивости."""

import numpy as np
import pytest

from credit_risk.metrics import gini_per_week, gini_stability, ks_statistic, score_panel


def _weekly_scores(weeks: np.ndarray, y: np.ndarray, strength, rng) -> np.ndarray:
    """Скоры, отделяющие классы с заданной силой. Ноль означает случайные."""
    strength = np.asarray(strength)[weeks] if np.ndim(strength) else strength
    return y * strength + rng.normal(0, 1, size=len(y))


def test_perfect_ranking_scores_one():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    assert score_panel(y, scores)["roc_auc"] == 1.0
    assert ks_statistic(y, scores) == pytest.approx(1.0)


def test_weeks_without_both_labels_are_skipped():
    weeks = np.array([1, 1, 2, 2])
    y = np.array([0, 0, 0, 1])  # в первой неделе только один класс

    scored_weeks, _ = gini_per_week(weeks, y, np.array([0.1, 0.2, 0.3, 0.9]))
    assert scored_weeks.tolist() == [2]


def test_stability_of_a_steady_model_is_its_mean_gini():
    # Недельные выборки намеренно большие. На малых выборках недельный джини
    # шумит, шум сам по себе даёт небольшой отрицательный наклон, а множитель 88
    # превращает этот шум в реальный штраф. Эта чувствительность и есть смысл
    # метрики, а не недостаток теста.
    rng = np.random.default_rng(7)
    weeks = np.repeat(np.arange(20), 3000)
    y = rng.binomial(1, 0.1, size=len(weeks))
    scores = _weekly_scores(weeks, y, 1.0, rng)

    _, ginis = gini_per_week(weeks, y, scores)

    assert gini_stability(weeks, y, scores) == pytest.approx(ginis.mean(), abs=0.05)


def test_a_decaying_model_is_punished():
    rng = np.random.default_rng(7)
    weeks = np.repeat(np.arange(20), 3000)
    y = rng.binomial(1, 0.1, size=len(weeks))

    # Разделяющая способность тает от сильной до нулевой к концу периода.
    scores = _weekly_scores(weeks, y, np.linspace(1.6, 0.0, 20), rng)
    _, ginis = gini_per_week(weeks, y, scores)
    stability = gini_stability(weeks, y, scores)

    assert stability < 0, "деградирующая модель должна уходить в минус"
    assert stability < ginis.mean() - 1.0, "штраф за наклон должен доминировать"
