"""Сколько шума в наших измерениях.

Весь проект построен на сравнении конфигураций, и разницы там бывают в тысячные
доли. Чтобы такие сравнения что-то значили, надо знать естественный разброс
самого измерения. Меряем его прямым повтором одной и той же конфигурации.

Источников случайности два, и их полезно разделить:

* **зерно модели** - LightGBM берёт случайные 80 процентов признаков на каждое
  дерево и случайную часть строк. Данные при этом одни и те же;
* **зерно выборки** - какие именно заявки попали в наши 600 тысяч.

Первый разброс показывает, насколько нестабильна модель. Сумма обоих показывает,
насколько нестабильно всё измерение целиком, и именно с ней надо сравнивать
разницы между конфигурациями.

    python scripts/repeatability.py --repeats 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.aggregates import attach_aggregates, attach_depth2  # noqa: E402
from credit_risk.config import REPORTS, SEED  # noqa: E402
from credit_risk.console import setup_console  # noqa: E402
from credit_risk.data import load_depth0  # noqa: E402
from credit_risk.features import build_features, to_model_frame  # noqa: E402
from credit_risk.validation import run_time_cv  # noqa: E402

DEPTH1 = ["applprev", "credit_bureau_a", "person"]
DEPTH2 = ["cb_a_pmts"]


def build(sample: int, seed: int):
    """Собрать матрицу признаков для заданного зерна выборки."""
    frame = load_depth0()
    if sample:
        frame = frame.sample(n=min(sample, frame.height), seed=seed)
    frame = attach_aggregates(frame, DEPTH1)
    frame = attach_depth2(frame, DEPTH2)
    return to_model_frame(build_features(frame))


def summarise(label: str, values: list[float]) -> dict:
    """Среднее, разброс и размах серии повторов."""
    array = np.array(values)
    return {
        "what": label,
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
        "spread": float(array.max() - array.min()),
    }


def main() -> None:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=600_000)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()

    started = time.time()
    seeds = [SEED + i for i in range(args.repeats)]

    # --- 1. Разное зерно модели, одни и те же данные ------------------------
    print("собираю признаки на фиксированной выборке...")
    x, y, weeks = build(args.sample, SEED)
    print(f"{len(x):,} заявок, {x.shape[1]} признаков\n")

    print("повторы при одном наборе данных, меняется только зерно модели:")
    model_auc, model_stab = [], []
    for seed in seeds:
        t0 = time.time()
        result = run_time_cv(
            x, y, weeks, n_splits=args.folds,
            params={"seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed},
        )
        model_auc.append(result.mean_auc)
        model_stab.append(result.stability)
        print(f"  зерно {seed}: AUC {result.mean_auc:.5f}  "
              f"устойчивость {result.stability:.5f}  ({time.time() - t0:.0f} с)")

    # --- 2. Разное зерно выборки и модели -----------------------------------
    print("\nповторы при разной выборке заявок:")
    both_auc, both_stab = [], []
    for seed in seeds:
        t0 = time.time()
        xs, ys, ws = build(args.sample, seed)
        result = run_time_cv(
            xs, ys, ws, n_splits=args.folds,
            params={"seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed},
        )
        both_auc.append(result.mean_auc)
        both_stab.append(result.stability)
        print(f"  зерно {seed}: AUC {result.mean_auc:.5f}  "
              f"устойчивость {result.stability:.5f}  ({time.time() - t0:.0f} с)")

    summary = [
        summarise("AUC, только зерно модели", model_auc),
        summarise("устойчивость, только зерно модели", model_stab),
        summarise("AUC, выборка и модель", both_auc),
        summarise("устойчивость, выборка и модель", both_stab),
    ]

    print(f"\n{'что меряем':38} {'среднее':>9} {'разброс':>9} {'размах':>9}")
    for row in summary:
        print(f"{row['what']:38} {row['mean']:>9.5f} {row['std']:>9.5f} {row['spread']:>9.5f}")

    auc_band = summary[2]["spread"]
    stab_band = summary[3]["spread"]
    print(f"\nвывод: разницы меньше {auc_band:.4f} по AUC и {stab_band:.4f} по "
          f"устойчивости не отличимы от шума при {args.repeats} повторах")

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "repeatability.json"
    out.write_text(
        json.dumps(
            {"sample": args.sample, "folds": args.folds, "seeds": seeds,
             "model_seed_only": {"auc": model_auc, "stability": model_stab},
             "sample_and_model": {"auc": both_auc, "stability": both_stab},
             "summary": summary},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nотчёт записан: {out}")
    print(f"всего времени: {time.time() - started:.0f} с")


if __name__ == "__main__":
    main()
