"""Проверки разбиения по времени и отбора колонок."""

import numpy as np
import polars as pl

from credit_risk.features import drop_useless_columns
from credit_risk.validation import expanding_window_splits


def test_training_never_sees_the_future():
    weeks = np.repeat(np.arange(40), 10)

    splits = list(expanding_window_splits(weeks, n_splits=4))
    assert len(splits) == 4

    for train_idx, valid_idx in splits:
        assert weeks[train_idx].max() < weeks[valid_idx].min()


def test_validation_blocks_do_not_overlap():
    weeks = np.repeat(np.arange(40), 10)

    seen: set[int] = set()
    for _, valid_idx in expanding_window_splits(weeks, n_splits=4):
        block = set(weeks[valid_idx].tolist())
        assert not (block & seen), "одна неделя не может проверяться дважды"
        seen |= block


def test_training_window_grows():
    weeks = np.repeat(np.arange(40), 10)

    sizes = [len(train_idx) for train_idx, _ in expanding_window_splits(weeks, n_splits=4)]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_useless_columns_are_dropped():
    # Порог по пустоте это 95 процентов, поэтому колонку надо сделать заметно
    # пустее, иначе тест проверял бы не то, что написано в коде.
    n = 100
    frame = pl.DataFrame(
        {
            "case_id": list(range(n)),
            "target": [i % 2 for i in range(n)],
            "WEEK_NUM": [i // 10 for i in range(n)],
            "constant_770L": [5] * n,
            "mostly_null_123A": [1.0] + [None] * (n - 1),
            "useful_456A": [float(i) for i in range(n)],
        }
    )

    kept = drop_useless_columns(frame).columns

    assert "useful_456A" in kept
    assert "constant_770L" not in kept, "постоянная колонка бесполезна"
    assert "mostly_null_123A" not in kept, "почти пустая колонка бесполезна"
    assert {"case_id", "target", "WEEK_NUM"} <= set(kept), "служебные колонки остаются"
