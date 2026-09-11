"""Вторая модель и ансамбль: LightGBM, CatBoost и их усреднение.

Смысл ансамбля не в том, что вторая модель лучше. Смысл в том, что две модели с
разным устройством ошибаются на разных заявителях. Там, где ошибки не совпадают,
усреднение их гасит. Если же модели предсказывают почти одно и то же, ансамбль
не даст ничего, и это видно заранее по корреляции их предсказаний.

Усреднение делается двумя способами. По вероятностям - обычное среднее. По
рангам - сначала каждая модель выстраивает заявки по порядку, потом усредняются
номера в очереди. Второй способ обычно лучше, когда метрика смотрит только на
порядок, а модели откалиброваны по-разному.

    python scripts/ensemble.py --sample 400000 --folds 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.aggregates import attach_aggregates, attach_depth2  # noqa: E402
from credit_risk.config import REPORTS, SEED  # noqa: E402
from credit_risk.console import setup_console  # noqa: E402
from credit_risk.data import load_depth0  # noqa: E402
from credit_risk.features import build_features, to_model_frame  # noqa: E402
from credit_risk.metrics import score_panel, stability_components  # noqa: E402
from credit_risk.model import prepare_for_catboost, train_fold, train_fold_catboost  # noqa: E402
from credit_risk.validation import time_splits  # noqa: E402

DEPTH1 = ["applprev", "credit_bureau_a", "credit_bureau_b", "person",
          "deposit", "debitcard", "other", "tax_a", "tax_b", "tax_c"]
DEPTH2 = ["cb_a_pmts", "cb_b_pmts", "applprev2", "person2"]


def main() -> None:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=400_000)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args()

    started = time.time()
    frame = load_depth0()
    if args.sample:
        frame = frame.sample(n=min(args.sample, frame.height), seed=SEED)
    frame = attach_aggregates(frame, DEPTH1)
    frame = attach_depth2(frame, DEPTH2)

    x, y, weeks = to_model_frame(build_features(frame))
    print(f"\nвыборка {len(x):,} заявок, {x.shape[1]} признаков "
          f"({time.time() - started:.0f} с)\n")

    lgb_oof = np.full(len(x), np.nan)
    cat_oof = np.full(len(x), np.nan)

    for fold, (train_idx, valid_idx) in enumerate(time_splits(weeks, args.folds), start=1):
        x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
        x_valid, y_valid = x.iloc[valid_idx], y.iloc[valid_idx]

        t0 = time.time()
        booster = train_fold(x_train, y_train, x_valid, y_valid)
        lgb_oof[valid_idx] = booster.predict(x_valid)
        t_lgb = time.time() - t0

        t0 = time.time()
        model = train_fold_catboost(x_train, y_train, x_valid, y_valid)
        cat_oof[valid_idx] = model.predict_proba(prepare_for_catboost(x_valid)[0])[:, 1]
        t_cat = time.time() - t0

        print(f"  фолд {fold}: LightGBM {t_lgb:>5.0f} с ({booster.num_trees()} деревьев), "
              f"CatBoost {t_cat:>5.0f} с ({model.tree_count_} деревьев)")

    seen = ~np.isnan(lgb_oof)
    yv, wv = y[seen].to_numpy(), weeks[seen]
    lgb_p, cat_p = lgb_oof[seen], cat_oof[seen]

    # Насколько две модели вообще расходятся. Если корреляция рангов близка к
    # единице, усреднять нечего: они видят заявителей одинаково.
    agreement = pd.Series(lgb_p).corr(pd.Series(cat_p), method="spearman")
    print(f"\nсогласие моделей по рангам: {agreement:.4f}")

    blends = {
        "LightGBM": lgb_p,
        "CatBoost": cat_p,
        "среднее вероятностей": (lgb_p + cat_p) / 2,
        "среднее рангов": (
            pd.Series(lgb_p).rank(pct=True).to_numpy()
            + pd.Series(cat_p).rank(pct=True).to_numpy()
        ) / 2,
    }

    print(f"\n{'модель':24} {'AUC':>8} {'джини':>8} {'устойчивость':>14}")
    rows = []
    for label, scores in blends.items():
        panel = score_panel(yv, scores)
        stability = stability_components(wv, yv, scores)["stability"]
        rows.append({
            "model": label, "auc": panel["roc_auc"],
            "gini": panel["gini"], "stability": stability,
        })
        print(f"{label:24} {panel['roc_auc']:>8.5f} {panel['gini']:>8.5f} {stability:>14.5f}")

    best_single = max(rows[:2], key=lambda r: r["auc"])
    print(f"\nлучшая одиночная модель: {best_single['model']}")
    for row in rows[2:]:
        print(f"  {row['model']:24} AUC {row['auc'] - best_single['auc']:+.5f}   "
              f"устойчивость {row['stability'] - best_single['stability']:+.5f}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "ensemble.json"
    out.write_text(
        json.dumps(
            {"rows_evaluated": int(seen.sum()), "rank_agreement": float(agreement),
             "folds": args.folds, "sample": args.sample, "runs": rows},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nотчёт записан: {out}")
    print(f"всего времени: {time.time() - started:.0f} с")


if __name__ == "__main__":
    main()
