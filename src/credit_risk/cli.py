"""Точка входа: одна команда, один воспроизводимый прогон."""

from __future__ import annotations

import argparse
import json
import time

from .config import REPORTS, SEED
from .console import setup_console
from .data import load_depth0
from .features import build_features, to_model_frame
from .validation import run_random_cv, run_time_cv


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучить бейзлайн на таблицах нулевой глубины.")
    parser.add_argument("--sample", type=int, default=None, help="взять только N заявок")
    parser.add_argument("--folds", type=int, default=4, help="число разбиений по времени")
    parser.add_argument("--tag", type=str, default="depth0", help="имя прогона для отчёта")
    parser.add_argument(
        "--compare-random",
        action="store_true",
        help="дополнительно прогнать случайную нарезку, чтобы увидеть завышение",
    )
    return parser.parse_args(argv)


def _print_result(title: str, result) -> None:
    print(f"\n--- {title} ---")
    print(f"AUC по фолдам: {[round(s, 5) for s in result.fold_scores]}")
    print(f"среднее {result.mean_auc:.5f}   разброс {result.std_auc:.5f}")
    if result.stability is not None:
        print(f"устойчивость (метрика соревнования): {result.stability:.5f}")
    for name, value in result.panel.items():
        print(f"  {name:26} {value:.5f}")


def main(argv: list[str] | None = None) -> None:
    setup_console()
    args = _parse_args(argv)

    started = time.time()
    frame = load_depth0()
    print(f"загружено: {frame.height:,} заявок, {frame.width} колонок "
          f"({time.time() - started:.0f} с)")

    if args.sample:
        frame = frame.sample(n=min(args.sample, frame.height), seed=SEED)
        print(f"выборка: {frame.height:,} заявок")

    frame = build_features(frame)
    x, y, weeks = to_model_frame(frame)
    print(f"признаков после отбора: {x.shape[1]}   доля дефолтов: {y.mean():.5f}")
    print(f"недель: {len(set(weeks))} (с {weeks.min()} по {weeks.max()})")

    time_result = run_time_cv(x, y, weeks, n_splits=args.folds)
    _print_result("разбиение по времени (честное)", time_result)

    report = {
        "tag": args.tag,
        "rows": int(len(x)),
        "features": int(x.shape[1]),
        "folds": args.folds,
        "time_cv": {
            "fold_auc": time_result.fold_scores,
            "mean_auc": time_result.mean_auc,
            "std_auc": time_result.std_auc,
            "stability": time_result.stability,
            **time_result.panel,
        },
    }

    if args.compare_random:
        random_result = run_random_cv(x, y, weeks)
        _print_result("случайная нарезка (завышает)", random_result)
        report["random_cv"] = {
            "fold_auc": random_result.fold_scores,
            "mean_auc": random_result.mean_auc,
            "stability": random_result.stability,
            **random_result.panel,
        }
        gap = random_result.mean_auc - time_result.mean_auc
        print(f"\nзавышение от случайной нарезки: {gap:+.5f} AUC")
        report["optimism_gap_auc"] = gap

    print("\nтоп признаков по вкладу")
    print(time_result.feature_importance.head(15).to_string(index=False))

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"{args.tag}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nотчёт записан: {out}")
    print(f"всего времени: {time.time() - started:.0f} с")


if __name__ == "__main__":
    main()
