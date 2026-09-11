"""Smoke tests: the pipeline runs and the metrics behave sensibly."""

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import CVConfig
from credit_risk.data import split_target
from credit_risk.features import build_features
from credit_risk.metrics import ks_statistic, score_panel
from credit_risk.validation import run_cv

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_synthetic import make_frame  # noqa: E402


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return make_frame(6000, seed=1)


def test_sentinel_is_removed(frame):
    features = build_features(frame)
    assert not (features["DAYS_EMPLOYED"] == 365243).any()


def test_domain_features_exist(frame):
    features = build_features(frame)
    for name in ("CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "AGE_YEARS"):
        assert name in features.columns


def test_perfect_ranking_scores_one():
    y = np.array([0, 0, 1, 1])
    assert score_panel(y, np.array([0.1, 0.2, 0.8, 0.9]))["roc_auc"] == 1.0
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_cv_beats_random(frame):
    x, y = split_target(frame)
    result = run_cv(build_features(x), y, CVConfig(n_splits=3))

    assert len(result.oof_predictions) == len(frame)
    assert result.mean_auc > 0.6, "the planted signal should be findable"
    assert result.std_auc < 0.1, "folds should broadly agree"
