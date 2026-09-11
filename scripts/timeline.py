"""Как менялся поток заявок и доля дефолтов по неделям.

Первое, что стоит посмотреть в любых данных с датой. Структурный слом в
популяции объясняет поведение метрик лучше, чем любые графики важности
признаков.

    python scripts/timeline.py --every 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.config import DATE_COLUMN, REPORTS, TARGET, WEEK_COLUMN  # noqa: E402
from credit_risk.console import setup_console  # noqa: E402
from credit_risk.data import load_base  # noqa: E402


def weekly_summary() -> pl.DataFrame:
    """Число заявок и доля дефолтов в каждой неделе."""
    return (
        load_base()
        .group_by(WEEK_COLUMN)
        .agg(
            pl.len().alias("applications"),
            pl.col(TARGET).mean().alias("default_rate"),
            pl.col(DATE_COLUMN).min().alias("week_start"),
        )
        .sort(WEEK_COLUMN)
    )


def main() -> None:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--every", type=int, default=4, help="печатать каждую N-ю неделю")
    args = parser.parse_args()

    summary = weekly_summary()
    median_volume = summary["applications"].median()

    print(f"{'нед':>4} {'дата':>12} {'заявок':>9} {'дефолт':>8}  {'от медианы':>10}")
    for row in summary.rows(named=True)[:: args.every]:
        share = row["applications"] / median_volume
        print(
            f"{row[WEEK_COLUMN]:>4} {str(row['week_start']):>12} "
            f"{row['applications']:>9,} {row['default_rate']:>8.4f}  {share:>9.0%}"
        )

    # Недели, где поток упал ниже половины обычного: там любая недельная метрика
    # считается по горстке дефолтов и потому ничего не значит.
    thin = summary.filter(pl.col("applications") < median_volume * 0.5)
    print(f"\nнедель с потоком ниже половины медианы: {thin.height}")
    if thin.height:
        print(f"  с {thin[WEEK_COLUMN].min()} по {thin[WEEK_COLUMN].max()}")
        print(f"  минимум: {thin['applications'].min():,} заявок")

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "timeline.json"
    out.write_text(
        json.dumps(
            {
                "median_weekly_applications": float(median_volume),
                "weeks": [
                    {
                        "week": r[WEEK_COLUMN],
                        "week_start": str(r["week_start"]),
                        "applications": r["applications"],
                        "default_rate": round(r["default_rate"], 5),
                    }
                    for r in summary.rows(named=True)
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nотчёт записан: {out}")


if __name__ == "__main__":
    main()
