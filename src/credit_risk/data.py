"""Loading the raw application table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_RAW, ID_COLUMN, TARGET

APPLICATION_TRAIN = "application_train.csv"

# Home Credit encodes "never employed" as this sentinel instead of a null.
# Leaving it in place lets the model learn a fake 1000-year employment history.
DAYS_EMPLOYED_SENTINEL = 365243


def load_application(path: Path | None = None, sample: int | None = None) -> pd.DataFrame:
    """Read the main application table.

    Args:
        path: csv to read. Defaults to ``data/raw/application_train.csv``.
        sample: if given, read only this many rows. Useful while iterating.
    """
    path = path or DATA_RAW / APPLICATION_TRAIN
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See README for how to fetch the data, "
            "or run scripts/make_synthetic.py for a runnable stand-in."
        )

    df = pd.read_csv(path, nrows=sample)
    if TARGET not in df.columns:
        raise ValueError(f"Column {TARGET!r} is missing from {path}")

    return df


def split_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Peel the label and the row id off the feature matrix."""
    y = df[TARGET].astype(int)
    x = df.drop(columns=[c for c in (TARGET, ID_COLUMN) if c in df.columns])
    return x, y
