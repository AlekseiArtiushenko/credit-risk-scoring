"""Подготовка признаков.

Осознанно небольшой модуль. Смысл бейзлайна в том, чтобы честно измерить, чего
стоят сырые колонки, до того как сверху наворачивается что-то умное.
"""

from __future__ import annotations

import polars as pl

from .config import DATE_COLUMN, ID_COLUMN, TARGET, WEEK_COLUMN

SERVICE_COLUMNS = (ID_COLUMN, TARGET, WEEK_COLUMN, DATE_COLUMN, "MONTH")

# Колонка, у которой почти все значения пустые или одно и то же, не несёт
# информации, но занимает память и замедляет обучение.
MAX_NULL_SHARE = 0.95
MIN_UNIQUE_VALUES = 2
MAX_CATEGORY_LEVELS = 200


def dates_to_offsets(df: pl.DataFrame) -> pl.DataFrame:
    """Заменить даты на число дней до даты решения по заявке.

    Абсолютная дата это ловушка: модель выучит конкретный календарный период и
    развалится на следующем. Относительный срок переносится во времени.
    """
    date_columns = [
        name
        for name, dtype in zip(df.columns, df.dtypes)
        if dtype == pl.Date and name != DATE_COLUMN
    ]
    if not date_columns:
        return df

    return df.with_columns(
        [
            (pl.col(DATE_COLUMN) - pl.col(name)).dt.total_days().cast(pl.Float32).alias(name)
            for name in date_columns
        ]
    )


def drop_useless_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Выбросить пустые, постоянные и слишком дробные категориальные колонки."""
    n_rows = df.height
    keep = []

    for name, dtype in zip(df.columns, df.dtypes):
        if name in SERVICE_COLUMNS:
            keep.append(name)
            continue

        column = df[name]
        if column.null_count() / n_rows > MAX_NULL_SHARE:
            continue
        n_unique = column.n_unique()
        if n_unique < MIN_UNIQUE_VALUES:
            continue
        if dtype == pl.String and n_unique > MAX_CATEGORY_LEVELS:
            continue

        keep.append(name)

    return df.select(keep)


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """Полное преобразование: сырая таблица на входе, матрица для модели на выходе."""
    return drop_useless_columns(dates_to_offsets(df))


def to_model_frame(df: pl.DataFrame):
    """Перевести в pandas и отделить метку, неделю и идентификатор от признаков.

    LightGBM понимает категориальный тип pandas напрямую, поэтому обходимся без
    one-hot и не раздуваем размерность.
    """
    pdf = df.to_pandas()

    y = pdf[TARGET].astype("int8")
    weeks = pdf[WEEK_COLUMN].to_numpy()
    x = pdf.drop(columns=[c for c in SERVICE_COLUMNS if c in pdf.columns])

    for name in x.select_dtypes(include=["object"]).columns:
        x[name] = x[name].astype("category")

    return x, y, weeks
