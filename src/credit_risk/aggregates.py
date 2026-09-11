"""Свёртка таблиц глубины один в признаки уровня заявки.

Модель смотрит на одну строку за раз. Таблицы глубины один устроены иначе: там
на одну заявку приходится столько строк, сколько у человека было прошлых
кредитов или обращений. Напрямую их приклеить нельзя, заявка размножится.

Поэтому многие строки схлопываются в одну: пять прошлых кредитов превращаются в
«сколько их было», «максимальная просрочка», «средняя сумма». Эти числа не
лежали ни в одной колонке, их не существовало, пока мы их не посчитали. Дерево
само так не умеет, потому что видит строку, а не группу строк.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .config import ID_COLUMN, PARQUET_TRAIN

# Какие таблицы глубины один умеем сворачивать и под каким коротким именем
# признаки попадут в итоговую таблицу.
DEPTH1_TABLES: dict[str, str] = {
    "applprev": "train_applprev_1_*",
    "credit_bureau_a": "train_credit_bureau_a_1_*",
    "credit_bureau_b": "train_credit_bureau_b_1",
    "person": "train_person_1",
    "deposit": "train_deposit_1",
    "debitcard": "train_debitcard_1",
    "other": "train_other_1",
    "tax_a": "train_tax_registry_a_1",
    "tax_b": "train_tax_registry_b_1",
    "tax_c": "train_tax_registry_c_1",
}

NUMERIC_SUFFIX = ("A", "P")
DATE_SUFFIX = ("D",)
CATEGORICAL_SUFFIX = ("M",)


def _column_plan(schema: dict[str, pl.DataType]) -> list[pl.Expr]:
    """Решить, какие свёртки посчитать для каждой колонки.

    Правило простое и опирается на суффикс имени. Для денег и просрочек имеет
    смысл максимум и среднее. Для дат имеет смысл самая свежая и самая старая.
    Для категорий имеет смысл только то, сколько их было разных.
    """
    expressions: list[pl.Expr] = [pl.len().alias("count")]

    for name, dtype in schema.items():
        if name == ID_COLUMN:
            continue

        letter = name[-1] if name else ""
        numeric = letter in NUMERIC_SUFFIX or dtype.is_numeric()

        if letter in DATE_SUFFIX:
            column = pl.col(name).cast(pl.Date, strict=False)
            expressions += [
                column.max().alias(f"{name}_last"),
                column.min().alias(f"{name}_first"),
            ]
        elif letter in CATEGORICAL_SUFFIX or dtype == pl.String:
            expressions.append(pl.col(name).n_unique().alias(f"{name}_nunique"))
        elif numeric:
            column = pl.col(name).cast(pl.Float32, strict=False)
            expressions += [
                column.max().alias(f"{name}_max"),
                column.mean().alias(f"{name}_mean"),
            ]

    return expressions


def aggregate_table(
    name: str, pattern: str | None = None, directory: Path | None = None
) -> pl.DataFrame:
    """Свернуть одну таблицу глубины один в одну строку на заявку."""
    directory = directory or PARQUET_TRAIN
    pattern = pattern or DEPTH1_TABLES[name]

    files = sorted(directory.glob(f"{pattern}.parquet"))
    if not files:
        raise FileNotFoundError(f"нет файлов {pattern}.parquet в {directory}")

    lazy = pl.scan_parquet(files)
    schema = dict(lazy.collect_schema())

    aggregated = lazy.group_by(ID_COLUMN).agg(_column_plan(schema)).collect()

    # Префикс не даёт колонкам из разных таблиц столкнуться именами и заодно
    # оставляет видимым, откуда признак взялся.
    return aggregated.rename(
        {c: f"{name}__{c}" for c in aggregated.columns if c != ID_COLUMN}
    )


def attach_aggregates(
    frame: pl.DataFrame, tables: list[str], directory: Path | None = None
) -> pl.DataFrame:
    """Приклеить свёртки перечисленных таблиц к таблице заявок."""
    for name in tables:
        aggregated = aggregate_table(name, directory=directory)
        before = frame.width
        frame = frame.join(aggregated, on=ID_COLUMN, how="left")

        # Покрытие считается по строкам итоговой таблицы, а не по размеру
        # свёртки: свёртка построена по всем заявкам, а frame может быть
        # выборкой. Пустая колонка count означает, что у заявителя записей нет.
        matched = frame[f"{name}__count"].is_not_null().sum()
        print(f"  {name:18} +{frame.width - before:>4} признаков "
              f"(есть история у {matched / frame.height:.0%} заявок)")

    return frame
