"""Детект хайлайтов: кандидаты + LLM -> финальный список моментов с названиями.

Если LLM нет, моменты всё равно формируются эвристикой: заголовок берётся из
самой «горячей» реплики или повторяющегося сообщения чата, категория — по
доминирующему сигналу.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from loguru import logger

from .asr import TranscriptSegment, transcript_excerpt
from .chat import ChatMessage, chat_excerpt, hype_score
from .config import Settings
from .features import Candidate
from .llm import LLMProvider, NullProvider, rank_moments
from .modes import chapter_line, fmt_timecode
from .utils import fmt_ts, slugify

CLUTCH_RE = re.compile(
    r"\b(ace|clutch|1v\d|2v\d|pentakill|penta|quadra|triple kill|headshot|"
    r"defus|ninja|колаут|клатч|пента|эйс|хедшот|вытащил|затащил)\b",
    re.IGNORECASE,
)
RAGE_RE = re.compile(
    r"\b(rage|furious|angry|тильт|тильтую|бесит|ярость|злюсь|ненавижу|заткн)\b",
    re.IGNORECASE,
)
FUNNY_RE = re.compile(
    r"\b(lol|lmao|haha|хаха|смешно|шутка|мем|прикол|ржу|ору)\b", re.IGNORECASE
)
STORY_RE = re.compile(
    r"\b(история|расскажу|случай|однажды|когда я|remember|story|so basically)\b",
    re.IGNORECASE,
)


@dataclass
class Moment:
    """Готовый хайлайт: что, где и почему."""

    index: int
    start: float
    end: float
    score: float
    title: str
    category: str = "other"
    reason: str = ""
    source: str = "heuristic"
    components: dict[str, float] = field(default_factory=dict)
    chat_excerpt: list[str] = field(default_factory=list)
    transcript: str = ""
    clip_path: str = ""
    clip_url: str = ""
    thumbnail_path: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def time_text(self) -> str:
        return fmt_ts(self.start)

    @property
    def slug(self) -> str:
        return slugify(self.title, max_len=50, fallback=f"moment_{self.index:02d}")

    def stamp(self, mode: str = "default") -> str:
        """Таймкод начала в оформлении мода ('2:04:19')."""
        return fmt_timecode(self.start, mode)

    def chapter(self, mode: str = "default") -> str:
        """Строка оглавления: '2:04:19 - Любимый цвет'."""
        return chapter_line(self.start, self.title, mode)

    def to_dict(self, mode: str = "default") -> dict[str, Any]:
        return {
            "index": self.index,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.duration, 2),
            "score": round(self.score, 2),
            "title": self.title,
            "category": self.category,
            "reason": self.reason,
            "source": self.source,
            "time": self.stamp(mode),
            "timecode": fmt_timecode(self.start, mode, with_ms=True),
            "chapter": self.chapter(mode),
            "components": {k: round(float(v), 2) for k, v in self.components.items()},
            "chat_excerpt": self.chat_excerpt[:20],
            "transcript": self.transcript,
            "clip_path": self.clip_path,
            "clip_url": self.clip_url,
            "thumbnail_path": self.thumbnail_path,
        }



def _pick_title(
    messages: Sequence[ChatMessage],
    segments: Sequence[TranscriptSegment],
    start: float,
    end: float,
    allow_chat_fallback: bool = True,
) -> str:
    """Заголовок момента — приоритет у РЕЧИ стримера, чат только как запасной вариант.

    Без расшифровки (или если в окне никто не говорил) честно сообщаем об этом,
    а не выдаём реплику зрителя за слова стримера.
    """
    best: tuple[float, str] = (0.0, "")
    for segment in segments:
        if segment.end < start or segment.start > end:
            continue
        text = segment.text.strip()
        if not (12 <= len(text) <= 120):
            continue
        words = len(text.split())
        if words < 3:  # отбросить «Мнмн», «Ааа», обрывки
            continue
        score, _, clip = hype_score(text)
        if CLUTCH_RE.search(text) or RAGE_RE.search(text):
            score += 3.0
        if clip:
            score += 2.0
        score += 0.25 * min(words, 20)
        if score > best[0]:
            best = (score, text)
    if best[0] >= 1.0:
        return _tidy(best[1])

    # Речи в окне нет — честная пометка вместо чужой реплики.
    has_speech = any(segment.end >= start and segment.start <= end for segment in segments)
    if not allow_chat_fallback and not has_speech:
        return f"Речь не распознана ({fmt_ts(start)})"
    window = [m.text.strip() for m in messages if start <= m.offset <= end and m.text.strip()]
    if window:
        common = Counter(window).most_common(1)[0]
        note = "чат" if not has_speech else "реплика зрителя"
        if common[1] >= 3:
            return f"«{_tidy(common[0], 40)}» ×{common[1]} ({note})"
        return f"«{_tidy(window[len(window) // 2], 40)}» ({note})"
    if best[1]:
        return _tidy(best[1])
    return f"Момент на {fmt_ts(start)}"


def _tidy(text: str, limit: int = 80) -> str:
    """Чистит реплику под заголовок: убирает хвосты, схлопывает пробелы."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
    if len(text) > limit:
        cut = text[:limit].rsplit(" ", 1)[0]
        text = (cut or text[:limit]).rstrip(",.;:-— ") + "…"
    return text



def _pick_category(
    components: dict[str, float],
    segments: Sequence[TranscriptSegment],
    start: float,
    end: float,
) -> str:
    text = transcript_excerpt(list(segments), start, end, max_chars=1200, use_translation=False)
    if CLUTCH_RE.search(text):
        return "clutch"
    if components.get("laughter", 0.0) > 0.5 and components.get("chat_hype", 0.0) > 0.3:
        return "funny"
    if RAGE_RE.search(text):
        return "rage"
    if FUNNY_RE.search(text):
        return "funny"
    if STORY_RE.search(text):
        return "story"
    if components.get("keywords", 0.0) >= 1.0 and components.get("keywords", 0.0) >= max(
        components.get("chat_rate", 0.0), components.get("audio_volume", 0.0)
    ):
        return "hype"
    if components.get("chat_clip", 0.0) > 0.5 or components.get("chat_rate", 0.0) > 1.5:
        return "reaction"
    if components.get("audio_dynamics", 0.0) > 1.5:
        return "hype"
    if components.get("chat_spam", 0.0) > 0.5:
        return "funny"
    return "other"


def _heuristic_reason(components: dict[str, float], messages: Sequence[ChatMessage], start: float, end: float) -> str:
    parts: list[str] = []
    if components.get("chat_rate", 0.0) > 0.5:
        parts.append(f"пик активности чата (z={components['chat_rate']:.1f})")
    if components.get("chat_clip", 0.0) > 0:
        clips = sum(1 for m in messages if start <= m.offset <= end and "clip" in m.text.lower())
        parts.append(f"просьбы заклипать ({clips})")
    if components.get("chat_hype", 0.0) > 0.5:
        parts.append(f"эмоут-спам (z={components['chat_hype']:.1f})")
    if components.get("audio_volume", 0.0) > 0.5:
        parts.append(f"громкая реакция (z={components['audio_volume']:.1f})")
    if components.get("audio_dynamics", 0.0) > 0.5:
        parts.append(f"резкий звук (z={components['audio_dynamics']:.1f})")
    if components.get("keywords", 0.0) > 0.5:
        parts.append(f"ключевые слова в речи (z={components['keywords']:.1f})")
    if components.get("laughter", 0.0) > 0.5:
        parts.append("смех в речи")
    if not parts:
        parts.append("локальный максимум совокупного сигнала")
    return "Сигналы: " + ", ".join(parts) + "."


def build_moments(
    candidates: Sequence[Candidate],
    messages: Sequence[ChatMessage],
    segments: Sequence[TranscriptSegment],
    settings: Settings,
    provider: LLMProvider | None = None,
    vod_title: str = "",
    language: str = "",
    duration: float | None = None,
    llm_progress: Any | None = None,
) -> list[Moment]:
    """Превращает кандидатов в готовые моменты (с LLM, если он есть)."""
    if not candidates:
        return []
    items: list[dict[str, Any]] = []
    max_candidates = min(len(candidates), int(settings.llm_max_candidates))
    for index, candidate in enumerate(candidates[:max_candidates]):
        items.append(
            {
                "index": index,
                "start": candidate.start,
                "end": candidate.end,
                "signals": candidate.components,
                "transcript": transcript_excerpt(
                    list(segments), candidate.start, candidate.end, max_chars=1600
                ),
                "chat": chat_excerpt(messages, candidate.start, candidate.end, limit=40),
            }
        )

    ratings: dict[int, Any] = {}
    if provider is not None and not isinstance(provider, NullProvider) and items:
        logger.info("LLM: оцениваем {} кандидатов (модель {})", len(items), provider.model)
        ratings = rank_moments(
            provider,
            items,
            vod_title=vod_title,
            language=language,
            batch=max(1, int(settings.llm_batch)),
        )
        if llm_progress is not None:
            llm_progress(len(ratings), len(items))

    peaks = [candidate.peak for candidate in candidates[:max_candidates]]
    peaks_sorted = sorted(peaks)
    top_peak = float(peaks[-1]) if peaks_sorted else 1.0
    median_peak = float(np.median(peaks_sorted)) if peaks_sorted else 0.0
    span = max(1e-6, top_peak - median_peak)

    moments: list[Moment] = []
    for index, candidate in enumerate(candidates):
        rating = ratings.get(index)
        # Относительная часть: ранг среди кандидатов этого VOD.
        relative = float(np.clip((candidate.peak - median_peak) / span, 0.0, 1.0))
        # Абсолютная часть: сила сигнала сама по себе (пик 6.0 = 100 баллов).
        absolute = float(np.clip(candidate.peak / 6.0, 0.0, 1.0))
        score = (0.55 * relative + 0.45 * absolute) * 100.0
        title = _pick_title(messages, segments, candidate.start, candidate.end)
        category = _pick_category(candidate.components, segments, candidate.start, candidate.end)
        reason = _heuristic_reason(
            candidate.components, messages, candidate.start, candidate.end
        )
        start, end = candidate.start, candidate.end
        source = "heuristic"
        if rating is not None:
            source = "llm+signals"
            weight = 0.65 if rating.title else 0.5
            score = (1.0 - weight) * score + weight * float(np.clip(rating.score, 0.0, 100.0))
            title = rating.title or title
            reason = rating.reason or reason
            if rating.category and rating.category != "other":
                category = rating.category
            if rating.start is not None and rating.end is not None:
                cand_start = max(0.0, rating.start)
                cand_end = max(cand_start + 5.0, rating.end)
                if 8.0 <= (cand_end - cand_start) <= settings.max_clip_length * 1.5:
                    start, end = cand_start, cand_end
        if duration:
            start = max(0.0, min(start, duration - 3.0))
            end = max(start + 5.0, min(end, duration))
        moments.append(
            Moment(
                index=index,
                start=start,
                end=end,
                score=score,
                title=title,
                category=category,
                reason=reason,
                source=source,
                components=dict(candidate.components),
                chat_excerpt=chat_excerpt(messages, start, end, limit=25),
                transcript=transcript_excerpt(
                    list(segments), start, end, max_chars=2500
                ),
            )
        )
    moments = select_moments(moments, settings)
    for position, moment in enumerate(moments, start=1):
        moment.index = position
    return moments


def select_moments(moments: Sequence[Moment], settings: Settings) -> list[Moment]:
    """Отбор моментов: дедупликация, добивание до ``min_moments``, потом срез по ``max_clips``."""
    ordered = sorted(moments, key=lambda item: (-item.score, item.start))
    want = max(0, int(settings.min_moments))
    cap = max(int(settings.max_clips), want)
    if want and cap == int(settings.max_clips):
        cap = max(cap, want)  # min_moments важнее max_clips

    chosen: list[Moment] = []
    # Первый круг: строгий разнос (половина min_gap) — самые сильные моменты.
    for factor in (1.0, 0.5, 0.0):
        if len(chosen) >= cap and not (want and len(chosen) < want):
            break
        gap = settings.min_gap * factor
        for moment in ordered:
            if len(chosen) >= cap:
                break
            if moment in chosen:
                continue
            if any(
                moment.start < other.end + gap and other.start < moment.end + gap
                for other in chosen
            ):
                continue
            chosen.append(moment)
        if want and len(chosen) >= want:
            break

    chosen.sort(key=lambda item: item.start)
    if want and len(chosen) < want:
        logger.warning(
            "Набрано только {} моментов вместо {}: сократите --min-gap (сейчас {:.0f} с) "
            "или --min-clip-length (сейчас {:.0f} с) — иначе клипы не влезают по времени.",
            len(chosen), want, settings.min_gap, settings.min_clip_length,
        )
    elif len(chosen) > cap:
        chosen = chosen[:cap]
        chosen.sort(key=lambda item: item.start)
    return chosen

