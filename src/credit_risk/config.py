"""Пути и константы проекта.

Всё, что может понадобиться поменять, лежит здесь, чтобы в остальном коде не
было разбросанных магических чисел.
"""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
REPORTS = PROJECT_ROOT / "reports"

# Распакованные таблицы соревнования.
PARQUET_TRAIN = DATA_RAW / "parquet_files" / "train"
PARQUET_TEST = DATA_RAW / "parquet_files" / "test"
FEATURE_DEFINITIONS = DATA_RAW / "feature_definitions.csv"

TARGET = "target"
ID_COLUMN = "case_id"
WEEK_COLUMN = "WEEK_NUM"
DATE_COLUMN = "date_decision"
SEED = 42


@dataclass(frozen=True)
class CVConfig:
    """Как устроена проверка модели на невиданных данных."""

    n_splits: int = 5
    shuffle: bool = True
    seed: int = SEED
