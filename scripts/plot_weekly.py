"""Нарисовать недельный джини и поток заявок на одной картинке.

Таблицы в README показывают средние. Картинка показывает то, чего в средних не
видно: где именно качество проседает и почему в этом месте ему нельзя верить.

Файл кладётся в docs/, а не в reports/, потому что reports/ не попадает в git, а
картинка нужна в README.

    python scripts/plot_weekly.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.config import PROJECT_ROOT, REPORTS  # noqa: E402
from credit_risk.console import setup_console  # noqa: E402

BREAK_WEEK = 64
OUT = PROJECT_ROOT / "docs" / "weekly-gini.svg"

# Фон задан явно белым: GitHub показывает картинку на своём фоне, и прозрачная
# подложка с тёмным текстом стала бы нечитаемой в тёмной теме.
BACKGROUND = "#ffffff"
INK = "#16202b"
MUTED = "#7b8c99"
LINE = "#14666d"
WARN = "#a63a25"


def main() -> None:
    setup_console()
    path = REPORTS / "error-analysis.json"
    if not path.exists():
        raise SystemExit(f"нет {path}, сначала запусти scripts/error_analysis.py")

    weekly = json.loads(path.read_text(encoding="utf-8"))["weekly"]
    weeks = np.array([r["week"] for r in weekly])
    ginis = np.array([r["gini"] for r in weekly])
    volume = np.array([r["applications"] for r in weekly])

    slope, intercept = np.polyfit(np.arange(len(ginis)), ginis, 1)
    trend = slope * np.arange(len(ginis)) + intercept
    thin = volume < np.median(volume) * 0.5

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]}, facecolor=BACKGROUND,
    )

    for axis in (top, bottom):
        axis.set_facecolor(BACKGROUND)
        axis.axvspan(BREAK_WEEK, weeks.max(), color=MUTED, alpha=0.08, zorder=0)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(MUTED)
        axis.tick_params(colors=MUTED, labelsize=9)

    top.plot(weeks, ginis, color=LINE, linewidth=1.6, marker="o", markersize=3.2,
             label="джини недели")
    top.plot(weeks, trend, color=INK, linewidth=1.2, linestyle="--",
             label=f"тренд, наклон {slope:+.4f} за неделю")
    top.scatter(weeks[thin], ginis[thin], s=58, facecolors="none", edgecolors=WARN,
                linewidths=1.6, zorder=5, label="поток ниже половины медианы")
    top.axvline(BREAK_WEEK, color=WARN, linewidth=1.1, linestyle=":")
    top.set_ylabel("джини", color=INK, fontsize=10)
    top.set_title("Качество модели по неделям и поток заявок", color=INK,
                  fontsize=12, loc="left", pad=12)
    top.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")

    bottom.fill_between(weeks, volume, color=LINE, alpha=0.22, zorder=1)
    bottom.plot(weeks, volume, color=LINE, linewidth=1.3, zorder=2)
    bottom.axvline(BREAK_WEEK, color=WARN, linewidth=1.1, linestyle=":")
    bottom.set_ylabel("заявок в неделю", color=INK, fontsize=10)
    bottom.set_xlabel("номер недели, 0 это начало января 2019", color=INK, fontsize=10)
    bottom.annotate(
        "ковидная пауза", xy=(68, volume[weeks == 68][0] if (weeks == 68).any() else 0),
        xytext=(70, volume.max() * 0.62), color=WARN, fontsize=9,
        arrowprops={"arrowstyle": "->", "color": WARN, "linewidth": 1},
    )

    figure.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Формат берётся из расширения: удобно рендерить png для быстрого просмотра.
    figure.savefig(OUT, facecolor=BACKGROUND, bbox_inches="tight")
    print(f"картинка записана: {OUT}")
    print(f"недель на графике: {len(weeks)}, из них с малым потоком: {thin.sum()}")


if __name__ == "__main__":
    main()
