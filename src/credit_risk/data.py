"""Загрузка таблиц соревнования.

Данные лежат в parquet и разложены по глубине вложенности. Глубина ноль это
одна строка на заявку, такие таблицы просто приклеиваются к базовой. Глубина
один и два дают много строк на заявку и требуют агрегации, они добавляются
отдельным шагом.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .config import DATE_COLUMN, ID_COLUMN, PARQUET_TRAIN

# Последняя буква имени колонки кодирует её тип. Соглашение придумали
# организаторы, и оно избавляет от ручного перебора двух сотен колонок.
SUFFIX_NUMERIC = ("P", "A")  # просрочка в днях и денежные суммы
SUFFIX_DATE = ("D",)
SUFFIX_CATEGORICAL = ("M",)  # замаскированные категории


def _apply_suffix_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    """Привести колонки к типам, которые обещает их имя."""
    casts = []
    for name, dtype in zip(df.columns, df.dtypes):
        if name in (ID_COLUMN, DATE_COLUMN) or len(name) < 2:
            continue
        letter = name[-1]

        if letter in SUFFIX_NUMERIC:
            casts.append(pl.col(name).cast(pl.Float32, strict=False))
        elif letter in SUFFIX_DATE:
            casts.append(pl.col(name).cast(pl.Date, strict=False))
        elif letter in SUFFIX_CATEGORICAL:
            casts.append(pl.col(name).cast(pl.String))
        elif dtype == pl.Boolean:
            casts.append(pl.col(name).cast(pl.Int8))

    return df.with_columns(casts) if casts else df


def load_table(pattern: str, directory: Path | None = None) -> pl.DataFrame:
    """Прочитать таблицу, собрав её из всех кусков.

    Крупные таблицы разрезаны на файлы вида ``train_static_0_0`` и
    ``train_static_0_1``. Для нас это одна таблица, поэтому склеиваем.
    """
    directory = directory or PARQUET_TRAIN
    files = sorted(directory.glob(f"{pattern}.parquet"))
    if not files:
        raise FileNotFoundError(
            f"не найдено ни одного файла {pattern}.parquet в {directory}. "
            "Данные качаются с Kaggle, инструкция в README."
        )

    frames = [_apply_suffix_dtypes(pl.read_parquet(f)) for f in files]
    return pl.concat(frames, how="diagonal_relaxed")


def load_base(directory: Path | None = None) -> pl.DataFrame:
    """Базовая таблица: заявка, дата решения, номер недели и метка дефолта."""
    directory = directory or PARQUET_TRAIN
    base = pl.read_parquet(directory / "train_base.parquet")
    return base.with_columns(pl.col(DATE_COLUMN).cast(pl.Date, strict=False))


def load_depth0(directory: Path | None = None) -> pl.DataFrame:
    """Базовая таблица вместе со всеми таблицами нулевой глубины.

    Это первый honest срез данных: ничего агрегировать не нужно, соединение
    один к одному, и уже получается больше двухсот признаков.
    """
    directory = directory or PARQUET_TRAIN
    frame = load_base(directory)

    for pattern in ("train_static_0_*", "train_static_cb_0"):
        table = load_table(pattern, directory)
        frame = frame.join(table, on=ID_COLUMN, how="left")

    return frame
