"""The gradient boosting model and its hyper-parameters."""

from __future__ import annotations

import lightgbm as lgb

from .config import SEED

# Conservative starting point. Slow learning rate plus early stopping is a
# better default than a tuned-looking set of numbers nobody can justify.
BASELINE_PARAMS: dict = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbosity": -1,
    "seed": SEED,
}

NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 100


def train_fold(
    x_train,
    y_train,
    x_valid,
    y_valid,
    params: dict | None = None,
) -> lgb.Booster:
    """Fit one fold with early stopping on its own validation slice."""
    params = {**BASELINE_PARAMS, **(params or {})}

    train_set = lgb.Dataset(x_train, label=y_train)
    valid_set = lgb.Dataset(x_valid, label=y_valid, reference=train_set)

    return lgb.train(
        params,
        train_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
