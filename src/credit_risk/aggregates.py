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


# ---------------------------------------------------------------------------
# Глубина два
# ---------------------------------------------------------------------------

# Таблицы глубины два дают много строк на каждую строку глубины один. В
# credit_bureau_a_2 это помесячная история платежей: num_group1 это номер
# кредита, num_group2 это номер месяца. 188 миллионов строк, в память целиком не
# лезет, поэтому считаем по файлам и складываем частичные итоги.
DEPTH2_TABLES: dict[str, str] = {
    "cb_a_pmts": "train_credit_bureau_a_2_*",
    "cb_b_pmts": "train_credit_bureau_b_2",
    "applprev2": "train_applprev_2",
    "person2": "train_person_2",
}

# Служебные колонки вложенности, признаками они не являются.
GROUP_COLUMNS = ("num_group1", "num_group2")


def _partial_expressions(columns: list[str]) -> list[pl.Expr]:
    """Частичные итоги по одному файлу.

    Складывать между файлами можно только такие величины, которые остаются
    верными при объединении групп: количество, сумма, максимум. Среднее так
    складывать нельзя, поэтому оно выводится в конце из суммы и количества.
    """
    expressions: list[pl.Expr] = [pl.len().alias("rows")]

    for name in columns:
        column = pl.col(name).cast(pl.Float32, strict=False)
        expressions += [
            column.max().alias(f"{name}::max"),
            column.sum().alias(f"{name}::sum"),
            column.count().alias(f"{name}::n"),
            (column > 0).sum().alias(f"{name}::positive"),
        ]

    return expressions


def _numeric_columns(schema: dict) -> list[str]:
    """Числовые колонки таблицы, кроме служебных номеров вложенности."""
    return [
        name
        for name, dtype in schema.items()
        if name not in GROUP_COLUMNS
        and name != ID_COLUMN
        and (dtype.is_numeric() or (name and name[-1] in NUMERIC_SUFFIX))
    ]


def aggregate_depth2(
    name: str, pattern: str | None = None, directory: Path | None = None
) -> pl.DataFrame:
    """Свернуть таблицу глубины два в одну строку на заявку, по файлам."""
    directory = directory or PARQUET_TRAIN
    pattern = pattern or DEPTH2_TABLES[name]

    files = sorted(directory.glob(f"{pattern}.parquet"))
    if not files:
        raise FileNotFoundError(f"нет файлов {pattern}.parquet в {directory}")

    schema = dict(pl.scan_parquet(files[0]).collect_schema())
    numeric = _numeric_columns(schema)

    if not numeric:
        # Таблица целиком категориальная, например адреса и работодатели.
        # Считать по ней имеет смысл только количество записей.
        partials = [
            pl.scan_parquet(f).group_by(ID_COLUMN).agg(pl.len().alias("rows")).collect()
            for f in files
        ]
        combined = pl.concat(partials).group_by(ID_COLUMN).agg(pl.col("rows").sum())
        return combined.rename({"rows": f"{name}__count"})

    partials = [
        pl.scan_parquet(f).group_by(ID_COLUMN).agg(_partial_expressions(numeric)).collect()
        for f in files
    ]

    # Один и тот же case_id может встретиться в нескольких файлах, поэтому после
    # склейки частичных итогов группируем ещё раз.
    merge = [pl.col("rows").sum().alias("rows")]
    for column in numeric:
        merge += [
            pl.col(f"{column}::max").max().alias(f"{column}::max"),
            pl.col(f"{column}::sum").sum().alias(f"{column}::sum"),
            pl.col(f"{column}::n").sum().alias(f"{column}::n"),
            pl.col(f"{column}::positive").sum().alias(f"{column}::positive"),
        ]

    combined = pl.concat(partials).group_by(ID_COLUMN).agg(merge)

    # Теперь выводим осмысленные признаки из складываемых величин.
    final = [pl.col("rows").alias(f"{name}__count")]
    for column in numeric:
        n = pl.col(f"{column}::n")
        # Деление на ноль там, где у заявителя записи есть, но эта колонка у всех
        # пустая, даёт NaN. NaN это не то же самое, что пропуск: пропуск означает
        # «неизвестно», а NaN просочился бы в статистики как число. Приводим к
        # пропуску явно.
        final += [
            pl.col(f"{column}::max").alias(f"{name}__{column}_max"),
            (pl.col(f"{column}::sum") / n).fill_nan(None).alias(f"{name}__{column}_mean"),
            # Доля месяцев с просрочкой информативнее, чем их абсолютное число:
            # у человека с длинной историей записей больше просто по возрасту.
            (pl.col(f"{column}::positive") / n)
            .fill_nan(None)
            .alias(f"{name}__{column}_share_positive"),
        ]

    return combined.select([pl.col(ID_COLUMN), *final])


def attach_depth2(
    frame: pl.DataFrame, tables: list[str], directory: Path | None = None
) -> pl.DataFrame:
    """Приклеить свёртки таблиц глубины два к таблице заявок."""
    for name in tables:
        aggregated = aggregate_depth2(name, directory=directory)
        before = frame.width
        frame = frame.join(aggregated, on=ID_COLUMN, how="left")
        matched = frame[f"{name}__count"].is_not_null().sum()
        print(f"  {name:18} +{frame.width - before:>4} признаков "
              f"(есть история у {matched / frame.height:.0%} заявок)")

    return frame
