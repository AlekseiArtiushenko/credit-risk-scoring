"""Command line entry point: one command, one reproducible run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import REPORTS, CVConfig
from .data import load_application, split_target
from .features import build_features
from .validation import run_cv


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the credit default baseline.")
    parser.add_argument("--data", type=Path, default=None, help="path to application_train.csv")
    parser.add_argument("--sample", type=int, default=None, help="read only N rows")
    parser.add_argument("--folds", type=int, default=5, help="number of CV folds")
    parser.add_argument("--tag", type=str, default="baseline", help="name for this run")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    raw = load_application(args.data, sample=args.sample)
    x, y = split_target(raw)
    x = build_features(x)

    print(f"rows={len(x):,}  features={x.shape[1]}  default rate={y.mean():.4f}")

    result = run_cv(x, y, CVConfig(n_splits=args.folds))

    print(f"\nfold AUC: {[round(s, 5) for s in result.fold_scores]}")
    print(f"mean {result.mean_auc:.5f}  std {result.std_auc:.5f}\n")
    for name, value in result.panel.items():
        print(f"{name:26} {value:.5f}")

    print("\ntop features by gain")
    print(result.feature_importance.head(15).to_string(index=False))

    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {
        "tag": args.tag,
        "rows": int(len(x)),
        "features": int(x.shape[1]),
        "folds": args.folds,
        "fold_auc": result.fold_scores,
        "mean_auc": result.mean_auc,
        "std_auc": result.std_auc,
        **result.panel,
    }
    out = REPORTS / f"{args.tag}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
