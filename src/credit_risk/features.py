"""Feature preparation for the application table.

Deliberately small. The point of a baseline is to be honest about what the raw
columns are worth before any clever engineering is layered on top.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DAYS_EMPLOYED_SENTINEL


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, turning division by zero into a null rather than an infinity."""
    return numerator / denominator.replace(0, np.nan)


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the handful of ratios a credit analyst would compute by hand.

    Affordability is a ratio question, not a level question: an annuity of
    twelve thousand means nothing until you divide it by income.
    """
    df = df.copy()

    if "DAYS_EMPLOYED" in df:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(DAYS_EMPLOYED_SENTINEL, np.nan)

    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"} <= set(df):
        df["CREDIT_INCOME_RATIO"] = _safe_ratio(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL"])

    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"} <= set(df):
        df["ANNUITY_INCOME_RATIO"] = _safe_ratio(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL"])

    if {"AMT_ANNUITY", "AMT_CREDIT"} <= set(df):
        # Inverse of the loan term in years: how fast the loan is repaid.
        df["CREDIT_TERM"] = _safe_ratio(df["AMT_ANNUITY"], df["AMT_CREDIT"])

    if {"AMT_GOODS_PRICE", "AMT_CREDIT"} <= set(df):
        # Above one means borrowing more than the asset is worth.
        df["CREDIT_GOODS_RATIO"] = _safe_ratio(df["AMT_CREDIT"], df["AMT_GOODS_PRICE"])

    if {"DAYS_EMPLOYED", "DAYS_BIRTH"} <= set(df):
        df["EMPLOYED_LIFE_RATIO"] = _safe_ratio(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])

    if "DAYS_BIRTH" in df:
        df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Cast object columns to pandas category dtype.

    LightGBM consumes that directly, so we avoid one-hot encoding and the
    dimensional blow-up that comes with it.
    """
    df = df.copy()
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].astype("category")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """The full baseline transform, raw table in, model matrix out."""
    return encode_categoricals(add_domain_features(df))
