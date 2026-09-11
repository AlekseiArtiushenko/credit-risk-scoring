"""Стоит ли отрезать «плохие» экономические периоды из обучения.

Все модели проверяются на одной и той же выборке, недели 78-91, то есть на
восстановлении после ковидной паузы. Меняется только то, на каких неделях
модель училась.

Сравнения, ради которых всё затевалось:

* «всё» против «только до ковида» - помогает ли кризисный период или мешает;
* «только до ковида» против «сдвинутое окно» - те же 64 недели, но другая
  эпоха. Одинаковый объём данных, поэтому разница это чистый эффект эпохи;
* «только кризис» - работает ли идея «учиться на похожем плохом периоде».

    python scripts/epoch_experiment.py --sample 1000000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.config import REPORTS, SEED  # noqa: E402
from credit_risk.console import setup_console  # noqa: E402
from credit_risk.data import load_depth0  # noqa: E402
from credit_risk.features import build_features, to_model_frame  # noqa: E402
from credit_risk.validation import run_splits  # noqa: E402

VALID_FROM, VALID_TO = 78, 91

# (имя, первая неделя обучения, последняя неделя обучения)
TRAINING_WINDOWS = [
    ("всё доступное",        0, 77),
    ("только до ковида",     0, 63),
    ("сдвинутое окно",      14, 77),
    ("только кризис",       64, 77),
]


def main() -> None:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=1_000_000)
    parser.add_argument("--min-week", type=int, default=1500)
    args = parser.parse_args()

    started = time.time()
    frame = load_depth0()
    if args.sample:
        frame = frame.sample(n=min(args.sample, frame.height), seed=SEED)

    x, y, weeks = to_model_frame(build_features(frame))
    print(f"выборка: {len(x):,} заявок, {x.shape[1]} признаков ({time.time() - started:.0f} с)")

    valid_mask = (weeks >= VALID_FROM) & (weeks <= VALID_TO)
    valid_idx = np.flatnonzero(valid_mask)
    print(f"проверка одна и та же для всех: недели {VALID_FROM}-{VALID_TO}, "
          f"{len(valid_idx):,} заявок, дефолтов {y.iloc[valid_idx].sum():,}\n")

    rows = []
    for name, first, last in TRAINING_WINDOWS:
        train_idx = np.flatnonzero((weeks >= first) & (weeks <= last))
        result = run_splits(x, y, weeks, [(train_idx, valid_idx)])

        rows.append(
            {
                "window": name,
                "weeks": f"{first}-{last}",
                "n_weeks": last - first + 1,
                "train_rows": int(len(train_idx)),
                "auc": result.panel["roc_auc"],
                "gini": result.panel["gini"],
                "stability": result.stability,
                "slope": result.components["slope"],
                "top_decile": result.panel["top_decile_default_rate"],
            }
        )
        print(f"  {name:20} недель {last - first + 1:>2}  "
              f"обучение {len(train_idx):>9,}  AUC {rows[-1]['auc']:.5f}  "
              f"джини {rows[-1]['gini']:.5f}  устойчивость {rows[-1]['stability']:.5f}")

    best = max(rows, key=lambda r: r["auc"])
    print(f"\nлучший AUC: «{best['window']}» — {best['auc']:.5f}")

    baseline = next(r for r in rows if r["window"] == "всё доступное")
    print("\nразница с обучением на всём доступном:")
    for r in rows:
        if r is baseline:
            continue
        print(f"  {r['window']:20} AUC {r['auc'] - baseline['auc']:+.5f}   "
              f"устойчивость {r['stability'] - baseline['stability']:+.5f}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "epoch-experiment.json"
    out.write_text(
        json.dumps(
            {"valid_weeks": [VALID_FROM, VALID_TO], "valid_rows": int(len(valid_idx)), "runs": rows},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nотчёт записан: {out}")
    print(f"всего времени: {time.time() - started:.0f} с")


if __name__ == "__main__":
    main()
