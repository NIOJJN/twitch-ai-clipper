"""Слияние сигналов (чат + звук + текст) в единую карту «интересности» и кандидаты.

Идея: каждый сигнал приводим к робастной z-оценке (медиана/MAD), складываем
с весами из настроек и ищем локальные максимумы — это кандидаты в хайлайты.
Дальше кандидатов может переупорядочить LLM (см. ``llm.py``), но базовый детект
работает и без модели.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from loguru import logger

from .asr import TranscriptSegment
from .chat import ChatSignals, hype_score
from .config import Settings

LAUGH_RE = re.compile(
    r"((\b\w?ha){2,}\b)|((\b\w?heh){2,}\b)|((\bа?ха){2,}\b)|(\(?\s*(смех|смеётся|ржёт|хихи))",
    re.IGNORECASE,
)
SCREAM_RE = re.compile(r"[A-ZА-Я]{4,}")
KEYWORD_RE = re.compile(
    r"\b(clip|no way|insane|unbelievable|impossible|godlike|cracked|goated|finally|"
    r"never|record|world record|clutch|ace|1v\d|pentakill|quadra|headshot|"
    r"офигеть|жесть|невозможно|рекорд|наконец|клатч|пента|хедшот|"
    r"да ладно|не может быть|как так|ничего себе)\b",
    re.IGNORECASE,
)


def robust_z(values: np.ndarray) -> np.ndarray:
    """Робастная z-оценка: (x - median) / (1.4826 * MAD). Нули там, где сигнал плоский."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if scale <= 1e-9:
        std = float(values.std())
        if std <= 1e-9:
            return np.zeros_like(values)
        scale = std
    z = (values - median) / scale
    return np.clip(z, -8.0, 12.0)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Скользящее среднее (для подавления одиночных выбросов)."""
    window = max(1, int(window))
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def resample_mean(values: np.ndarray, src_hop: float, bucket: float, length: int) -> np.ndarray:
    """Усредняет per-second массив к бакетам длиной ``bucket``."""
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros(length, dtype=np.float64)
    if values.size == 0 or length == 0:
        return out
    cumulative = np.concatenate([[0.0], np.cumsum(values)])
    src_hop = max(1e-6, float(src_hop))
    for i in range(length):
        low = int((i * bucket) / src_hop)
        high = int(np.ceil(((i + 1) * bucket) / src_hop))
        low = max(0, min(low, values.size - 1))
        high = max(low + 1, min(high, values.size))
        out[i] = (cumulative[high] - cumulative[low]) / (high - low)
    return out


@dataclass
class Candidate:
    """Кандидат в хайлайт."""

    start: float
    end: float
    score: float
    peak: float
    components: dict[str, float] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "score": round(self.score, 3),
            "peak": round(self.peak, 3),
            "components": {k: round(float(v), 3) for k, v in self.components.items()},
        }


@dataclass
class SignalTable:
    """Карта интересности по таймлайну."""

    bucket: float
    times: np.ndarray
    score: np.ndarray
    components: dict[str, np.ndarray]

    def window_components(self, start: float, end: float) -> dict[str, float]:
        if self.times.size == 0:
            return {}
        mask = (self.times >= start) & (self.times <= end)
        if not mask.any():
            mask = self.times >= start
        result = {}
        for name, values in self.components.items():
            chunk = values[mask]
            result[name] = float(chunk.max()) if chunk.size else 0.0
        return result



def transcript_scores(
    segments: Sequence[TranscriptSegment],
    times: np.ndarray,
    bucket: float,
) -> dict[str, np.ndarray]:
    """По-бакетные текстовые сигналы: хайп-слова, смех, крик, ключевые фразы."""
    length = len(times)
    keywords = np.zeros(length)
    laughter = np.zeros(length)
    screaming = np.zeros(length)
    if not segments or length == 0:
        return {"keywords": keywords, "laughter": laughter, "screaming": screaming}
    for segment in segments:
        slot = int(np.clip(segment.start / bucket, 0, length - 1))
        text = segment.text or ""
        if not text:
            continue
        keywords[slot] += len(KEYWORD_RE.findall(text))
        laughter[slot] += len(LAUGH_RE.findall(text))
        if re.search(r"\[(смех|laughter)\]", text, re.IGNORECASE):
            laughter[slot] += 2
        screaming[slot] += len(SCREAM_RE.findall(text))
        hype, _, _ = hype_score(text)
        keywords[slot] += 0.5 * hype
    return {"keywords": keywords, "laughter": laughter, "screaming": screaming}


def build_signals(
    chat: ChatSignals,
    settings: Settings,
    rms_db: np.ndarray | None = None,
    dynamics: np.ndarray | None = None,
    segments: Sequence[TranscriptSegment] | None = None,
    duration: float | None = None,
) -> SignalTable:
    """Собирает единый скор: z-оценки сигналов с весами из настроек."""
    bucket = float(settings.bucket)
    span = float(duration or chat.duration or 0.0)
    if span <= 0:
        span = 1.0
    length = max(1, int(np.ceil(span / bucket)))
    times = (np.arange(length, dtype=np.float64) + 0.5) * bucket

    def fit(values: np.ndarray) -> np.ndarray:
        """Подгоняет длину ряда чата (он может не покрывать всю длительность VOD)."""
        resized = np.zeros(length)
        if values.size:
            take = min(length, values.size)
            resized[:take] = values[:take]
        return resized

    raw_rate = smooth(fit(chat.rate), 2)
    raw_hype = smooth(fit(chat.hype), 2)
    raw_clip = fit(chat.clip_requests)
    raw_unique = smooth(fit(chat.unique), 3)
    raw_spam = smooth(fit(chat.spam), 3)

    component_arrays: dict[str, np.ndarray] = {
        "chat_rate": raw_rate,
        "chat_hype": raw_hype,
        "chat_clip": raw_clip,
        "chat_unique": raw_unique,
        "chat_spam": raw_spam,
    }
    if rms_db is not None and rms_db.size:
        component_arrays["audio_volume"] = resample_mean(rms_db, 1.0, bucket, length)
    if dynamics is not None and dynamics.size:
        component_arrays["audio_dynamics"] = resample_mean(dynamics, 1.0, bucket, length)
    if segments:
        component_arrays.update(transcript_scores(segments, times, bucket))

    weights = {
        "chat_rate": settings.w_chat_rate,
        "chat_hype": settings.w_chat_hype,
        "chat_clip": 1.6,
        "chat_unique": settings.w_chat_unique,
        "chat_spam": settings.w_chat_spam,
        "audio_volume": settings.w_audio,
        "audio_dynamics": settings.w_audio * 0.8,
        "keywords": settings.w_keywords,
        "laughter": settings.w_laughter,
        "screaming": settings.w_laughter * 0.6,
    }

    normalized: dict[str, np.ndarray] = {}
    score = np.zeros(length, dtype=np.float64)
    for name, values in component_arrays.items():
        z = robust_z(values)
        normalized[name] = z
        score += weights.get(name, 0.0) * z

    positive = raw_rate[raw_rate > 0]
    if positive.size:
        score += 0.25 * np.clip(raw_rate / max(1e-6, float(np.median(positive))), 0.0, 2.0)
    score = smooth(score, 2)
    return SignalTable(bucket=bucket, times=times, score=score, components=normalized)



def _max_candidates_possible(table: SignalTable, settings: Settings) -> int:
    """Сколько клипов физически помещается в таймлайн при текущем min_gap/длине клипа."""
    total = float(table.times[-1] + table.bucket) if table.times.size else 0.0
    step = max(1.0, settings.min_clip_length + settings.min_gap)
    if total <= 0 or step <= 0:
        return 0
    return int(total // step)


def pick_candidates(table: SignalTable, settings: Settings) -> list[Candidate]:
    """Локальные максимумы с подавлением соседей — кандидаты в клипы.

    Стратегия в три прохода, чтобы набрать ``settings.min_moments``:
      1) строгий  — только score > 0, разнос min_gap (самые сильные моменты);
      2) мягкий   — любые локальные максимумы, прежний разнос;
      3) плотный  — равномерный разнос по таймлайну, если моментов всё ещё мало.
    """
    if table.score.size == 0:
        return []
    bucket = table.bucket
    min_len = max(1, int(round(settings.min_clip_length / bucket)))
    max_len = max(min_len, int(round(settings.max_clip_length / bucket)))
    want = max(0, int(settings.min_moments))
    hard_cap = max(int(settings.top_candidates), want + 10, int(settings.max_clips))

    def sweep(
        suppression: int,
        score_floor: float,
        limit: int,
        order: Any = None,
        taken: Any = None,
        collected: list[Candidate] | None = None,
    ) -> tuple[list[Candidate], Any]:
        """Один проход по отсортированным окнам с подавлением соседей."""
        bucket_times = table.times
        if order is None:
            order = np.argsort(-table.score)
        if taken is None:
            taken = np.zeros(table.score.size, dtype=bool)
        if collected is None:
            collected = []
        for index in order:
            if len(collected) >= limit:
                break
            if table.score[index] <= score_floor:
                continue
            if taken[max(0, index - suppression) : index + suppression + 1].any():
                continue
            left = right = int(index)
            threshold = 0.45 * float(table.score[index])
            while left - 1 >= 0 and table.score[left - 1] >= threshold and (right - left) < max_len:
                left -= 1
            while (
                right + 1 < table.score.size
                and table.score[right + 1] >= threshold
                and (right - left) < max_len
            ):
                right += 1
            while (right - left) < min_len:
                if left > 0:
                    left -= 1
                elif right + 1 < table.score.size:
                    right += 1
                else:
                    break
            start = max(0.0, left * bucket)
            end = min((right + 1) * bucket, bucket_times[-1] + bucket)
            collected.append(
                Candidate(
                    start=start,
                    end=end,
                    score=float(table.score[index]),
                    peak=float(table.score[index]),
                    components=table.window_components(start, end),
                )
            )
            taken[max(0, index - suppression) : index + suppression + 1] = True
        return collected, taken

    strict_gap = max(1, int(round(settings.min_gap / bucket)))
    candidates, taken = sweep(strict_gap, 0.0, hard_cap)

    if len(candidates) < want:
        soft_gap = max(1, strict_gap // 2)
        candidates, taken = sweep(soft_gap, -1e9, want + 5, taken=taken, collected=candidates)

    if len(candidates) < want:
        # Равномерный разнос: берём лучшие окна с фиксированным шагом.
        dense_gap = max(1, int(round(max(10.0, settings.min_clip_length) / bucket)))
        order = np.argsort(-table.score)
        candidates, taken = sweep(dense_gap, -1e9, want + 5, order=order, taken=taken, collected=candidates)

    candidates.sort(key=lambda item: (-item.score, item.start))
    if candidates:
        logger.info(
            "Кандидатов: {} (хотели {}) | макс. возможно при min_gap={:.0f}с и длине {}с: ~{}",
            len(candidates), want, settings.min_gap, settings.min_clip_length,
            _max_candidates_possible(table, settings),
        )
    return candidates
