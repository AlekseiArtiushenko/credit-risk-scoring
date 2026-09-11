"""Generate a small stand-in for application_train.csv.

The real data sits behind a Kaggle competition login. This script builds a file
with the same column names, dtypes and quirks so the pipeline can be run and
tested by anyone who clones the repo. It is a smoke-test fixture, not data to
draw conclusions from.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 0
CONTRACT_TYPES = ["Cash loans", "Revolving loans"]
EDUCATION = ["Secondary", "Higher education", "Incomplete higher", "Lower secondary"]
FAMILY = ["Married", "Single / not married", "Civil marriage", "Widow", "Separated"]
INCOME_TYPE = ["Working", "Commercial associate", "Pensioner", "State servant"]


def make_frame(n_rows: int, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    income = rng.lognormal(mean=11.9, sigma=0.5, size=n_rows).round(-2)
    credit = income * rng.lognormal(mean=0.9, sigma=0.6, size=n_rows)
    annuity = credit * rng.uniform(0.03, 0.12, size=n_rows)
    days_birth = -rng.integers(7500, 25000, size=n_rows)
    days_employed = -rng.integers(0, 12000, size=n_rows).astype(float)

    # Reproduce the sentinel: pensioners carry 365243 instead of a null.
    pensioner = rng.random(n_rows) < 0.18
    days_employed[pensioner] = 365243

    ext_source_2 = rng.beta(2, 2, size=n_rows)
    ext_source_3 = rng.beta(2, 2, size=n_rows)

    df = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100001, 100001 + n_rows),
            "NAME_CONTRACT_TYPE": rng.choice(CONTRACT_TYPES, n_rows, p=[0.9, 0.1]),
            "CODE_GENDER": rng.choice(["M", "F"], n_rows),
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit.round(-2),
            "AMT_ANNUITY": annuity.round(-1),
            "AMT_GOODS_PRICE": (credit * rng.uniform(0.8, 1.0, n_rows)).round(-2),
            "NAME_INCOME_TYPE": rng.choice(INCOME_TYPE, n_rows),
            "NAME_EDUCATION_TYPE": rng.choice(EDUCATION, n_rows),
            "NAME_FAMILY_STATUS": rng.choice(FAMILY, n_rows),
            "DAYS_BIRTH": days_birth,
            "DAYS_EMPLOYED": days_employed,
            "CNT_CHILDREN": rng.poisson(0.4, n_rows),
            "EXT_SOURCE_2": ext_source_2,
            "EXT_SOURCE_3": ext_source_3,
        }
    )

    # A genuine signal the model should be able to find, plus noise.
    logit = (
        -3.6  # tuned so the fixture lands near the real 8 percent base rate
        - 2.0 * (ext_source_2 - 0.5)
        - 1.8 * (ext_source_3 - 0.5)
        + 0.45 * np.log1p(df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"])
        + 0.35 * (df["NAME_EDUCATION_TYPE"] == "Lower secondary")
        + 0.9 * (days_birth / -25000)
        + rng.normal(0, 0.4, n_rows)
    )
    probability = 1 / (1 + np.exp(-logit))
    df["TARGET"] = (rng.random(n_rows) < probability).astype(int)

    # Real files have holes. A pipeline that only works on complete data is a toy.
    for column in ("AMT_ANNUITY", "AMT_GOODS_PRICE", "EXT_SOURCE_3"):
        missing = rng.random(n_rows) < 0.05
        df.loc[missing, column] = np.nan

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument(
        "--out", type=Path, default=Path("data/raw/application_train.csv")
    )
    args = parser.parse_args()

    frame = make_frame(args.rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"wrote {len(frame):,} rows to {args.out} (default rate {frame.TARGET.mean():.4f})")


if __name__ == "__main__":
    main()
