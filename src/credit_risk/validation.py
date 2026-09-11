"""Out-of-fold cross-validation.

Every number this project reports comes from predictions made on rows the model
did not see during fitting. That is the whole discipline in one sentence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import CVConfig
from .metrics import score_panel
from .model import train_fold


@dataclass
class CVResult:
    """Everything produced by a cross-validation run."""

    oof_predictions: np.ndarray
    fold_scores: list[float]
    feature_importance: pd.DataFrame
    panel: dict[str, float] = field(default_factory=dict)

    @property
    def mean_auc(self) -> float:
        return float(np.mean(self.fold_scores))

    @property
    def std_auc(self) -> float:
        """Spread across folds. A large spread means the mean is not trustworthy."""
        return float(np.std(self.fold_scores))


def run_cv(x: pd.DataFrame, y: pd.Series, config: CVConfig | None = None) -> CVResult:
    """Fit one model per fold and collect out-of-fold predictions."""
    config = config or CVConfig()
    splitter = StratifiedKFold(
        n_splits=config.n_splits, shuffle=config.shuffle, random_state=config.seed
    )

    oof = np.zeros(len(x))
    fold_scores: list[float] = []
    importances = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y), start=1):
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

    return CVResult(
        oof_predictions=oof,
        fold_scores=fold_scores,
        feature_importance=importance,
        panel=score_panel(y, oof),
    )
