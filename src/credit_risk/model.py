"""Модель градиентного бустинга и её гиперпараметры."""

from __future__ import annotations

import lightgbm as lgb

from .config import SEED

# Осторожная отправная точка. Медленная скорость обучения плюс ранняя остановка
# лучше, чем подогнанный набор чисел, который никто не может обосновать.
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
    """Обучить одну модель с ранней остановкой по своей проверочной части."""
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


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------

# Нужен не ради прироста от смены алгоритма, а ради ансамбля: две модели с
# разным устройством ошибаются на разных заявителях, и усреднение гасит ту часть
# ошибки, которая у них не совпадает.
CATBOOST_PARAMS: dict = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "learning_rate": 0.06,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "random_seed": SEED,
    "allow_writing_files": False,
    "verbose": False,
}

CATBOOST_ITERATIONS = 1500
CATBOOST_EARLY_STOPPING = 100

# CatBoost не принимает пропуски в категориальных колонках: для него категория
# это строка, а пропуск строкой не является. Заменяем явной меткой, чтобы
# «неизвестно» осталось отдельным значением, а не смешалось с чем-то ещё.
MISSING_CATEGORY = "__нет значения__"


def prepare_for_catboost(x):
    """Привести категориальные колонки к виду, который понимает CatBoost."""
    x = x.copy()
    categorical = [c for c in x.columns if str(x[c].dtype) == "category"]

    for name in categorical:
        x[name] = x[name].astype("object").fillna(MISSING_CATEGORY).astype(str)

    return x, categorical


def train_fold_catboost(x_train, y_train, x_valid, y_valid, params: dict | None = None):
    """Обучить одну модель CatBoost с ранней остановкой."""
    from catboost import CatBoostClassifier, Pool

    params = {**CATBOOST_PARAMS, **(params or {})}

    x_train, categorical = prepare_for_catboost(x_train)
    x_valid, _ = prepare_for_catboost(x_valid)

    train_pool = Pool(x_train, y_train, cat_features=categorical)
    valid_pool = Pool(x_valid, y_valid, cat_features=categorical)

    model = CatBoostClassifier(iterations=CATBOOST_ITERATIONS, **params)
    model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=CATBOOST_EARLY_STOPPING)
    return model
