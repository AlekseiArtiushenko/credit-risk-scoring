"""Заглянуть внутрь parquet-таблицы.

Паркет это бинарный формат, текстовым редактором его не открыть. Скрипт
показывает размер таблицы, типы колонок и несколько первых строк.

    python scripts/peek.py train_base
    python scripts/peek.py train_static_0_0 --rows 5 --columns 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.config import FEATURE_DEFINITIONS, PARQUET_TRAIN  # noqa: E402
from credit_risk.console import setup_console  # noqa: E402


def main() -> None:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", help="имя файла без расширения, например train_base")
    parser.add_argument("--rows", type=int, default=5, help="сколько строк показать")
    parser.add_argument("--columns", type=int, default=12, help="сколько колонок показать")
    parser.add_argument("--describe", action="store_true", help="добавить описания колонок")
    args = parser.parse_args()

    path = PARQUET_TRAIN / f"{args.table}.parquet"
    if not path.exists():
        available = sorted(p.stem for p in PARQUET_TRAIN.glob("*.parquet"))
        print(f"нет такого файла: {path}\n\nдоступные таблицы:")
        for name in available:
            print(" ", name)
        raise SystemExit(1)

    lazy = pl.scan_parquet(path)
    schema = lazy.collect_schema()
    n_rows = lazy.select(pl.len()).collect().item()

    print(f"таблица: {args.table}")
    print(f"строк: {n_rows:,}   колонок: {len(schema.names())}   "
          f"размер файла: {path.stat().st_size / 1e6:.0f} МБ")

    shown = schema.names()[: args.columns]
    print(f"\nпервые {len(shown)} колонок и {args.rows} строк:")
    with pl.Config(tbl_cols=len(shown), tbl_width_chars=200, fmt_str_lengths=24):
        print(lazy.select(shown).head(args.rows).collect())

    if args.describe and FEATURE_DEFINITIONS.exists():
        definitions = pl.read_csv(FEATURE_DEFINITIONS)
        subset = definitions.filter(pl.col("Variable").is_in(shown))
        print("\nописания колонок:")
        for variable, description in subset.iter_rows():
            print(f"  {variable:34} {description}")


if __name__ == "__main__":
    main()
