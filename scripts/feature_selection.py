"""Чистка признаков: что можно выбросить и сколько это стоит.

Определить «на глаз», какая колонка шум, нельзя. Единственный честный способ это
выбросить пачку и посмотреть, упало ли качество. Скрипт делает это для нескольких
стратегий отбора и сравнивает их с полным набором на одном разбиении.

Стратегии:

* **нулевой вклад** - признаки, по которым модель ни разу не сделала разбиение.
  Они не участвуют в предсказании вообще, выбрасываются бесплатно.
* **дубликаты** - пары признаков с корреляцией выше порога. Из пары остаётся тот,
  у кого выше вклад: второй несёт то же самое.
* **топ-N** - оставить только N самых весомых. Грубо, зато показывает, сколько
  признаков реально работает.

    python scripts/feature_selection.py --sample 800000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.aggregates import attach_aggregates, attach_depth2  # noqa: E402
from credit_risk.config import REPORTS, SEED  # noqa: E402
from credit_risk.console import setup_console  # noqa: E402
from credit_risk.data import load_depth0  # noqa: E402
from credit_risk.features import build_features, to_model_frame  # noqa: E402
from credit_risk.validation import run_time_cv  # noqa: E402

DEPTH1 = ["applprev", "credit_bureau_a", "credit_bureau_b", "person",
          "deposit", "debitcard", "other", "tax_a", "tax_b", "tax_c"]
DEPTH2 = ["cb_a_pmts", "cb_b_pmts", "applprev2", "person2"]

CORRELATION_THRESHOLD = 0.99
CORRELATION_SAMPLE = 100_000


def duplicate_features(
    x: pd.DataFrame, importance: pd.DataFrame, threshold: float, seed: int = SEED
) -> list[str]:
    """Найти пары почти одинаковых признаков и вернуть менее ценный из каждой.

    Корреляция считается на подвыборке строк: на всей матрице это дорого, а
    оценка от этого практически не меняется.
    """
    numeric = x.select_dtypes(include=[np.number])
    if len(numeric) > CORRELATION_SAMPLE:
        numeric = numeric.sample(CORRELATION_SAMPLE, random_state=seed)

    matrix = numeric.corr().abs()
    rank = dict(zip(importance["feature"], importance["gain"]))

    # Смотрим только верхний треугольник, чтобы не обрабатывать пару дважды.
    upper = matrix.where(np.triu(np.ones(matrix.shape), k=1).astype(bool))

    drop: set[str] = set()
    for column in upper.columns:
        partners = upper.index[upper[column] > threshold]
        for other in partners:
            if other in drop or column in drop:
                continue
            # Оставляем тот, которым модель пользуется активнее.
            drop.add(other if rank.get(other, 0) <= rank.get(column, 0) else column)

    return sorted(drop)


def main() -> None:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=800_000)
    parser.add_argument("--folds", type=int, default=4)
    args = parser.parse_args()

    started = time.time()
    frame = load_depth0()
    if args.sample:
        frame = frame.sample(n=min(args.sample, frame.height), seed=SEED)
    frame = attach_aggregates(frame, DEPTH1)
    frame = attach_depth2(frame, DEPTH2)

    x, y, weeks = to_model_frame(build_features(frame))
    print(f"\nполный набор: {len(x):,} заявок, {x.shape[1]} признаков "
          f"({time.time() - started:.0f} с)\n")

    runs: list[dict] = []

    def evaluate(label: str, columns: list[str]):
        """Обучить на заданном наборе колонок и записать строку результата."""
        started_run = time.time()
        result = run_time_cv(x[columns], y, weeks, n_splits=args.folds)
        row = {
            "strategy": label,
            "features": len(columns),
            "auc": result.mean_auc,
            "stability": result.stability,
            "gini": result.panel["gini"],
            "seconds": round(time.time() - started_run),
        }
        runs.append(row)
        print(f"  {label:26} признаков {len(columns):>4}  AUC {row['auc']:.5f}  "
              f"устойчивость {row['stability']:.5f}  ({row['seconds']} с)")
        return row, result

    print("прогоны:")
    # Полный набор обучается один раз: из этого же прогона берём и вклад
    # признаков, который определяет, кого считать лишним.
    full, full_result = evaluate("полный набор", list(x.columns))

    importance = full_result.feature_importance
    used = importance[importance["gain"] > 0]["feature"].tolist()
    dead = sorted(set(x.columns) - set(used))
    print(f"\nпризнаков с нулевым вкладом: {len(dead)} из {x.shape[1]}")

    if dead:
        evaluate("без нулевого вклада", used)

    duplicates = duplicate_features(x, importance, CORRELATION_THRESHOLD)
    print(f"пар-дубликатов с корреляцией выше {CORRELATION_THRESHOLD}: {len(duplicates)}")
    if duplicates:
        evaluate("без дубликатов", [c for c in x.columns if c not in set(duplicates)])

    for n in (250, 150, 75):
        if n < len(used):
            evaluate(f"топ-{n} по вкладу", importance.head(n)["feature"].tolist())

    print("\nразница с полным набором:")
    for row in runs:
        if row is full:
            continue
        print(f"  {row['strategy']:26} AUC {row['auc'] - full['auc']:+.5f}   "
              f"устойчивость {row['stability'] - full['stability']:+.5f}   "
              f"признаков {row['features'] - full['features']:+d}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "feature-selection.json"
    out.write_text(
        json.dumps(
            {"zero_gain": len(dead), "duplicates": len(duplicates), "runs": runs},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nотчёт записан: {out}")
    print(f"всего времени: {time.time() - started:.0f} с")


if __name__ == "__main__":
    main()
