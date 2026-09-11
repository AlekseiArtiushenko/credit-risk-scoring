"""Project-wide paths and constants.

Everything that a reader might want to change lives here, so the rest of the
code has no magic numbers scattered through it.
"""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
REPORTS = PROJECT_ROOT / "reports"

TARGET = "TARGET"
ID_COLUMN = "SK_ID_CURR"
SEED = 42


@dataclass(frozen=True)
class CVConfig:
    """How the out-of-fold validation is built."""

    n_splits: int = 5
    shuffle: bool = True
    seed: int = SEED
