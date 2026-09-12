"""Выровнять markdown-таблицы в README, чтобы они читались в исходнике.

На отрисованную страницу это не влияет вообще: markdown выравнивание игнорирует.
Влияет на того, кто открывает файл в редакторе, а таких у README большинство.

    python scripts/format_tables.py README.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.console import setup_console  # noqa: E402

SEPARATOR_CHARS = set("-: ")


def split_row(line: str) -> list[str]:
    """Разобрать строку таблицы на ячейки."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_separator(cells: list[str]) -> bool:
    """Строка из дефисов и двоеточий, отделяющая заголовок от тела."""
    return bool(cells) and all(cell and set(cell) <= SEPARATOR_CHARS for cell in cells)


def alignment(cell: str) -> tuple[bool, bool]:
    """Маркеры выравнивания в разделителе: двоеточие слева и справа."""
    return cell.startswith(":"), cell.endswith(":")


def format_table(rows: list[str]) -> list[str] | None:
    """Выровнять один блок. Возвращает None, если блок не похож на таблицу."""
    cells = [split_row(row) for row in rows]

    if len(cells) < 2 or not is_separator(cells[1]):
        return None
    if len({len(row) for row in cells}) != 1:
        # Разное число колонок: скорее всего внутри ячейки есть символ «|».
        # Такое лучше не трогать, чем сломать.
        return None

    n = len(cells[0])
    widths = [max(len(row[i]) for row in cells) for i in range(n)]
    marks = [alignment(cell) for cell in cells[1]]

    out = []
    for index, row in enumerate(cells):
        if index == 1:
            parts = []
            for i, (left, right) in enumerate(marks):
                inner = ("-" * widths[i])
                if left:
                    inner = ":" + inner[1:]
                if right:
                    inner = inner[:-1] + ":"
                parts.append(inner)
        else:
            # Содержимое центруется: так таблица читается как таблица, а не как
            # лесенка, и совпадает со стилем уже выровненных вручную блоков.
            parts = [row[i].center(widths[i]) for i in range(n)]

        out.append("| " + " | ".join(parts) + " |")

    return out


def format_document(text: str) -> tuple[str, int]:
    """Пройти по документу и выровнять все найденные таблицы."""
    lines = text.split("\n")
    result: list[str] = []
    block: list[str] = []
    changed = 0

    def flush() -> None:
        nonlocal changed
        if not block:
            return
        formatted = format_table(block)
        if formatted is None:
            result.extend(block)
        else:
            if formatted != block:
                changed += 1
            result.extend(formatted)
        block.clear()

    for line in lines:
        if line.lstrip().startswith("|"):
            block.append(line)
        else:
            flush()
            result.append(line)
    flush()

    return "\n".join(result), changed


def main() -> None:
    setup_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", default=["README.md"], type=Path)
    args = parser.parse_args()

    for path in args.files:
        text = path.read_text(encoding="utf-8")
        formatted, changed = format_document(text)
        path.write_text(formatted, encoding="utf-8")
        print(f"{path}: выровнено таблиц {changed}")


if __name__ == "__main__":
    main()
