"""Разбор ошибок модели: по неделям, по сегментам заявителей и по калибровке.

Итоговая цифра говорит, насколько модель хороша в среднем. Разбор говорит, на
ком именно она ошибается, а это уже разговор, который можно вести с бизнесом.

    python scripts/error_analysis.py --sample 800000
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

from credit_risk.aggregates import attach_aggregates  # noqa: E402
from credit_risk.config import REPORTS, SEED  # noqa: E402
from credit_risk.console import setup_console  # noqa: E402
from credit_risk.data import load_depth0  # noqa: E402
from credit_risk.features import build_features, to_model_frame  # noqa: E402
from credit_risk.metrics import gini, gini_per_week  # noqa: E402
from credit_risk.validation import run_time_cv  # noqa: E402

TABLES = ["applprev", "credit_bureau_a", "person"]
BREAK_WEEK = 64  # конец марта 2020, ковидная пауза
MIN_SEGMENT = 500


def _gini_or_nan(y: pd.Series, score: np.ndarray) -> float:
    """Джини сегмента, либо пусто, если сегмент слишком мал или однороден."""
    if len(y) < MIN_SEGMENT or y.nunique() < 2:
        return float("nan")
    return gini(y.to_numpy(), score)


def _segments(x: pd.DataFrame) -> dict[str, pd.Series]:
    """Разрезы заявителей, по которым смотрим качество отдельно."""
    out: dict[str, pd.Series] = {}

    if "credit_bureau_a__count" in x:
        has = x["credit_bureau_a__count"].notna()
        out["история в бюро есть"] = has
        out["истории в бюро нет"] = ~has

    if "applprev__count" in x:
        has = x["applprev__count"].notna()
        out["обращался раньше"] = has
        out["обращается впервые"] = ~has

    if "dateofbirth_337D" in x:
        age = x["dateofbirth_337D"] / 365.25
        out["возраст до 30"] = age < 30
        out["возраст 30-45"] = (age >= 30) & (age < 45)
        out["возраст 45-60"] = (age >= 45) & (age < 60)
        out["возраст 60 и старше"] = age >= 60

    if "credit_bureau_a__count" in x:
        n = x["credit_bureau_a__count"]
        out["кредитов в бюро 1-3"] = n.between(1, 3)
        out["кредитов в бюро 4-10"] = n.between(4, 10)
        out["кредитов в бюро 11 и больше"] = n >= 11

    return out


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
    frame = attach_aggregates(frame, TABLES)

    x, y, weeks = to_model_frame(build_features(frame))
    print(f"выборка {len(x):,} заявок, {x.shape[1]} признаков ({time.time() - started:.0f} с)\n")

    result = run_time_cv(x, y, weeks, n_splits=args.folds)
    seen = result.evaluated
    xv, yv = x[seen], y[seen]
    pv, wv = result.oof_predictions[seen], weeks[seen]

    print(f"оценено {seen.sum():,} заявок на неделях {wv.min()}-{wv.max()}")
    print(f"AUC {result.panel['roc_auc']:.4f}   джини {result.panel['gini']:.4f}\n")

    report: dict = {"rows_evaluated": int(seen.sum()), "panel": result.panel}

    # --- 1. До слома и после -------------------------------------------------
    print("=== период ===")
    print(f"{'период':22} {'заявок':>9} {'дефолтов':>9} {'джини':>7}")
    periods = {"до недели 64": wv < BREAK_WEEK, "с недели 64": wv >= BREAK_WEEK}
    report["periods"] = {}
    for label, mask in periods.items():
        g = _gini_or_nan(yv[mask], pv[mask])
        print(f"{label:22} {mask.sum():>9,} {yv[mask].mean():>9.4f} {g:>7.4f}")
        report["periods"][label] = {
            "rows": int(mask.sum()), "default_rate": float(yv[mask].mean()), "gini": g
        }

    # --- 2. Сегменты заявителей ---------------------------------------------
    print("\n=== сегменты заявителей ===")
    print(f"{'сегмент':30} {'заявок':>9} {'дефолтов':>9} {'джини':>7}  отличие")
    overall = result.panel["gini"]
    report["segments"] = {}
    for label, mask in _segments(xv).items():
        mask = mask.fillna(False).to_numpy() if hasattr(mask, "fillna") else mask
        if mask.sum() < MIN_SEGMENT:
            continue
        g = _gini_or_nan(yv[mask], pv[mask])
        mark = "" if np.isnan(g) else f"{g - overall:+.4f}"
        print(f"{label:30} {mask.sum():>9,} {yv[mask].mean():>9.4f} {g:>7.4f}  {mark}")
        report["segments"][label] = {
            "rows": int(mask.sum()), "default_rate": float(yv[mask].mean()), "gini": g
        }

    # --- 3. Калибровка -------------------------------------------------------
    print("\n=== калибровка: обещано против случившегося ===")
    print(f"{'дециль по скору':18} {'обещано':>9} {'случилось':>10} {'ошибка':>9}")
    deciles = pd.qcut(pv, 10, labels=False, duplicates="drop")
    report["calibration"] = []
    for d in range(int(deciles.max()) + 1):
        mask = deciles == d
        promised, actual = float(pv[mask].mean()), float(yv[mask].mean())
        ratio = promised / actual if actual > 0 else float("nan")
        print(f"{'дециль ' + str(d + 1):18} {promised:>9.4f} {actual:>10.4f} {ratio:>8.2f}x")
        report["calibration"].append(
            {"decile": d + 1, "predicted": promised, "actual": actual, "ratio": ratio}
        )

    overall_ratio = float(pv.mean()) / float(yv.mean())
    print(f"\nв целом модель обещает {pv.mean():.4f}, случается {yv.mean():.4f} "
          f"({overall_ratio:.2f}x)")
    report["calibration_overall"] = {
        "predicted": float(pv.mean()), "actual": float(yv.mean()), "ratio": overall_ratio
    }

    # --- 4. Недельный джини и поток -----------------------------------------
    week_numbers, week_ginis = gini_per_week(wv, yv.to_numpy(), pv)
    counts = pd.Series(wv).value_counts().sort_index()
    report["weekly"] = [
        {"week": int(w), "gini": float(g), "applications": int(counts.get(w, 0))}
        for w, g in zip(week_numbers, week_ginis)
    ]

    thin = [r for r in report["weekly"] if r["applications"] < counts.median() * 0.5]
    print(f"\nнедель с потоком ниже половины медианы: {len(thin)}")
    if thin:
        thin_g = np.mean([r["gini"] for r in thin])
        fat_g = np.mean([r["gini"] for r in report["weekly"] if r not in thin])
        print(f"  средний джини в них {thin_g:.4f}, в остальных {fat_g:.4f}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "error-analysis.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nотчёт записан: {out}")
    print(f"всего времени: {time.time() - started:.0f} с")


if __name__ == "__main__":
    main()
