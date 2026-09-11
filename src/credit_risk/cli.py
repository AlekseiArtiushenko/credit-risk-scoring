"""Точка входа: одна команда, один воспроизводимый прогон."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .aggregates import DEPTH1_TABLES, DEPTH2_TABLES, attach_aggregates, attach_depth2
from .config import REPORTS, SEED
from .console import setup_console
from .data import load_depth0
from .features import build_features, to_model_frame
from .metrics import stability_components
from .validation import run_holdout, run_random_cv, run_time_cv


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Обучить бейзлайн на таблицах нулевой глубины.")
    parser.add_argument("--sample", type=int, default=None, help="взять только N заявок")
    parser.add_argument("--folds", type=int, default=4, help="число разбиений по времени")
    parser.add_argument("--tag", type=str, default="depth0", help="имя прогона для отчёта")
    parser.add_argument(
        "--window",
        choices=("expanding", "sliding"),
        default="expanding",
        help="растущее окно обучения или скользящее постоянной ширины",
    )
    parser.add_argument(
        "--holdout-from",
        type=int,
        default=None,
        help="одна отсечка: учимся до этой недели, проверяемся на всём, что после",
    )
    parser.add_argument(
        "--min-week",
        type=int,
        default=2000,
        help="для справки: порог по числу заявок, ниже которого неделя считается пустой",
    )
    parser.add_argument(
        "--aggregates",
        nargs="*",
        default=[],
        metavar="TABLE",
        choices=list(DEPTH1_TABLES),
        help="свернуть и приклеить таблицы глубины один: " + ", ".join(DEPTH1_TABLES),
    )
    parser.add_argument(
        "--depth2",
        nargs="*",
        default=[],
        metavar="TABLE",
        choices=list(DEPTH2_TABLES),
        help="свернуть и приклеить таблицы глубины два: " + ", ".join(DEPTH2_TABLES),
    )
    parser.add_argument(
        "--compare-random",
        action="store_true",
        help="дополнительно прогнать случайную нарезку, чтобы увидеть завышение",
    )
    return parser.parse_args(argv)


def _print_components(title: str, k: dict) -> None:
    print(f"  {title}: {k['stability']:.5f}")
    print(f"    средний джини по неделям  {k['mean_gini']:+.5f}   недель {len(k['weeks'])}")
    print(f"    наклон за неделю          {k['slope']:+.5f}  -> штраф {k['slope_penalty']:+.5f}")
    print(f"    разброс вокруг прямой     {k['residual_std']:+.5f}  -> штраф {k['residual_penalty']:+.5f}")


def _print_result(title: str, result) -> None:
    print(f"\n--- {title} ---")
    print(f"AUC по фолдам: {[round(s, 5) for s in result.fold_scores]}")
    if len(result.fold_scores) > 1:
        print(f"среднее {result.mean_auc:.5f}   разброс {result.std_auc:.5f}")
    if result.components is not None:
        _print_components("устойчивость, все недели", result.components)
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

    if args.aggregates:
        print()
        print("свёртки таблиц глубины один:")
        frame = attach_aggregates(frame, args.aggregates)
        print(f"колонок после свёрток: {frame.width}")

    if args.depth2:
        print()
        print("свёртки таблиц глубины два:")
        frame = attach_depth2(frame, args.depth2)
        print(f"колонок после свёрток: {frame.width}")

    frame = build_features(frame)
    x, y, weeks = to_model_frame(frame)
    print(f"признаков после отбора: {x.shape[1]}   доля дефолтов: {y.mean():.5f}")
    print(f"недель: {len(set(weeks))} (с {weeks.min()} по {weeks.max()})")

    report = {
        "tag": args.tag,
        "rows": int(len(x)),
        "features": int(x.shape[1]),
        "aggregates": args.aggregates,
        "depth2": args.depth2,
    }

    if args.holdout_from is not None:
        train_n = int((weeks < args.holdout_from).sum())
        valid_n = int((weeks >= args.holdout_from).sum())
        print(f"\nотсечка на неделе {args.holdout_from}: "
              f"обучение {train_n:,} заявок, проверка {valid_n:,}")

        result = run_holdout(x, y, weeks, args.holdout_from)
        _print_result(f"одна отсечка по времени, с недели {args.holdout_from}", result)

        # Справочный вариант: недели с мизерным потоком выкинуты. Соревнование так
        # не считает, но без этого разброс определяется неделями ковидной паузы.
        evaluated = result.evaluated
        clean = stability_components(
            weeks[evaluated],
            y[evaluated].to_numpy(),
            result.oof_predictions[evaluated],
            min_count=args.min_week,
        )
        print()
        _print_components(f"устойчивость, недели от {args.min_week} заявок", clean)

        report["holdout"] = {
            "holdout_from": args.holdout_from,
            "train_rows": train_n,
            "valid_rows": valid_n,
            "auc": result.fold_scores[0],
            "stability_all_weeks": result.stability,
            "components": result.components,
            "stability_dense_weeks": clean["stability"],
            "components_dense_weeks": clean,
            **result.panel,
        }
        importance = result.feature_importance
    else:
        result = run_time_cv(x, y, weeks, n_splits=args.folds, window=args.window)
        _print_result(f"разбиение по времени, окно {args.window}", result)
        report["folds"] = args.folds
        report["window"] = args.window
        report["time_cv"] = {
            "fold_auc": result.fold_scores,
            "mean_auc": result.mean_auc,
            "std_auc": result.std_auc,
            "stability": result.stability,
            "components": result.components,
            **result.panel,
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
            gap = random_result.mean_auc - result.mean_auc
            print(f"\nзавышение от случайной нарезки: {gap:+.5f} AUC")
            report["optimism_gap_auc"] = gap

        importance = result.feature_importance

    print("\nтоп признаков по вкладу")
    print(importance.head(15).to_string(index=False))

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"{args.tag}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nотчёт записан: {out}")
    print(f"всего времени: {time.time() - started:.0f} с")


if __name__ == "__main__":
    main()
