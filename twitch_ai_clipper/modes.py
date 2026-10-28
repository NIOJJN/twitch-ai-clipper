"""Моды (профили) работы: раскладка детекта + оформление таймкодов.

Мод ``toparachevt`` (алиас ``топарашевт``) повторяет оглавление стрима
Топарашевта — таймкод всегда с часами, затем название через ``" - "``::

    2:04:19 - Любимый цвет
    2:06:03 - Корм для мужчин
    2:07:57 - Нравится ли быть стримерами?

Под ``toparachevt`` раскладка детекта плотная: глава примерно каждые две
минуты стрима (в эталонном оглавлении 125 глав на 4 ч 45 мин, минимальный
зазор между ними — 18 с). Заголовки глав должен придумывать LLM, поэтому мод
поднимает ``llm_max_candidates`` (иначе назовутся только первые 32 главы).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from .utils import fmt_ts

DEFAULT_MODE = "default"


@dataclass(frozen=True)
class Mode:
    """Один мод: как искать моменты и как их потом показывать."""

    name: str
    title: str
    description: str
    force_hours: bool = False          # таймкод всегда 'Ч:ММ:СС' (мод toparachevt)
    separator: str = " - "             # между таймкодом и названием
    chapter_list: bool = False         # печатать/писать оглавление главой
    title_max_len: int = 0             # 0 = не обрезать название
    detect: dict[str, Any] = field(default_factory=dict)

    def stamp(self, seconds: float, with_ms: bool = False) -> str:
        """Секунды -> таймкод в оформлении мода."""
        return fmt_ts(seconds, with_ms=with_ms, force_hours=self.force_hours)

    def line(self, seconds: float, title: str) -> str:
        """Одна строка оглавления: ``2:04:19 - Любимый цвет``."""
        return f"{self.stamp(seconds)}{self.separator}{shorten(title, self.title_max_len)}"


MODES: dict[str, Mode] = {
    DEFAULT_MODE: Mode(
        name=DEFAULT_MODE,
        title="Стандартный",
        description="Как раньше: HH:MM:SS (MM:SS до первого часа), таблица моментов",
    ),
    "toparachevt": Mode(
        name="toparachevt",
        title="Топарашевт",
        description=(
            "Оглавление стрима таймкодами: '2:04:19 - Любимый цвет' "
            "(часы всегда, глава примерно каждые 2 минуты)"
        ),
        force_hours=True,
        separator=" - ",
        chapter_list=True,
        title_max_len=80,
        detect={
            # ~1 глава на 2 минуты стрима — плотность эталонного оглавления.
            "max_clips": 300,
            "min_moments": 150,
            "min_gap": 60.0,
            "top_candidates": 1500,
            "bucket": 5.0,
            "min_clip_length": 10.0,
            "max_clip_length": 90.0,
            "clip_length": 30.0,
            # Глав много, значит LLM должен назвать их все, а не только первые 32.
            "llm_max_candidates": 200,
            "llm_batch": 6,
        },
    ),
}

# Кириллица и частые опечатки в названии канала -> каноническое имя мода.
ALIASES: dict[str, str] = {
    "топарашевт": "toparachevt",
    "топарашефт": "toparachevt",
    "toparashet": "toparachevt",
    "toparashevt": "toparachevt",
}

_WHITESPACE_RE = re.compile(r"\s+")


def shorten(title: str, limit: int) -> str:
    """Схлопывает пробелы и, если надо, аккуратно обрезает название по слову."""
    text = _WHITESPACE_RE.sub(" ", (title or "").strip())
    if limit <= 0 or len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return (cut or text[:limit]).rstrip(",.;:-— ") + "…"


def mode_name(name: str | None) -> str:
    """Люмое написание мода -> каноническое имя (пустая строка = дефолт)."""
    text = (name or "").strip().lower()
    if not text:
        return DEFAULT_MODE
    return ALIASES.get(text, text)


def resolve_mode(name: str | None) -> Mode:
    """Мод по имени; неизвестное имя — ошибка со списком доступных."""
    canonical = mode_name(name)
    mode = MODES.get(canonical)
    if mode is None:
        raise ValueError(
            f"Неизвестный мод: {name!r}. Доступные: {', '.join(sorted(MODES))}"
            + " (алиас: топарашевт)"
        )
    return mode


def get_mode(name: str | None) -> Mode:
    """Мод по имени; неизвестное имя молча превращается в дефолтный."""
    canonical = mode_name(name)
    return MODES.get(canonical, MODES[DEFAULT_MODE])


def apply_mode(name: str | None, overrides: dict[str, Any]) -> dict[str, Any]:
    """Кладёт настройки мода в overrides (явные флаги CLI и --preset важнее)."""
    if not (name or "").strip():
        return overrides
    mode = resolve_mode(name)
    for key, value in mode.detect.items():
        if key not in overrides:
            overrides[key] = value
    overrides["mode"] = mode.name
    return overrides


def fmt_timecode(seconds: float, mode: str | None = DEFAULT_MODE, with_ms: bool = False) -> str:
    """Таймкод в оформлении мода."""
    return get_mode(mode).stamp(seconds, with_ms=with_ms)


def chapter_line(seconds: float, title: str, mode: str | None = DEFAULT_MODE) -> str:
    """Строка оглавления в оформлении мода: '2:04:19 - Любимый цвет'."""
    return get_mode(mode).line(seconds, title)


def chapter_lines(
    items: Sequence[tuple[float, str]], mode: str | None = DEFAULT_MODE
) -> list[str]:
    """Оглавление целиком по парам (секунды, название)."""
    resolved = get_mode(mode)
    return [resolved.line(start, title) for start, title in items]