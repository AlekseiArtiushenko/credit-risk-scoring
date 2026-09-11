"""Настройка вывода в консоль.

На русской Windows консоль по умолчанию работает в кодировке cp1251, и любая
попытка напечатать кириллицу или рамки таблиц падает с UnicodeEncodeError.
Переключаем поток вывода на UTF-8 до того, как что-либо печатаем.
"""

from __future__ import annotations

import sys


def setup_console() -> None:
    """Перевести stdout и stderr на UTF-8, если они на чём-то другом."""
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", "") or "").lower()
        if encoding.replace("-", "") != "utf8":
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                # Поток перенаправлен куда-то, где так нельзя. Не критично.
                pass
