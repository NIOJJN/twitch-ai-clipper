"""Анализ чата: агрегация по окнам, эмоут-профиль, всплески активности.

Чат — самый дешёвый и очень точный индикатор интересных моментов: пик сообщений,
эмоут-спам, просьбы «clip it» почти всегда совпадают с хайлайтом.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .twitch_api import ChatMessage
from .utils import read_jsonl

# Эмоуты/мемы, которые зрители используют именно на «событиях».
HYPE_EMOTES: tuple[str, ...] = (
    "LUL", "LULW", "KEKW", "KekW", "OMEGALUL", "Pog", "PogChamp", "PogU", "Poggers",
    "POGGERS", "PogO", "PogTasty", "LETSGO", "LETSGOPOG", "ICANT", "Clueless", "WaitWhat",
    "monkaS", "monkaW", "Sadge", "PepeHands", "widepeepoSad", "peepoSad", "peepoHappy",
    "peepoArrive", "peepoLeave", "WidePeepoHappy", "AYAYA", "HYPERS", "catJAM", "catKISS",
    "GIGACHAD", "EZ", "EZClap", "EZY", "Clap", "CLEAN", "NOOO", "NOOOO", "4Head", "Jebaited",
    "xqcL", "WutFace", "KomodoHype", "PJSalt", "TriHard", "Kreygasm", "SourPls", "FeelsGoodMan",
    "FeelsBadMan", "FeelsOkayMan", "NODDERS", "Copium", "HOPIUM", "Bedge", "BasedGod", "susge",
    "Susge", "FORGOR", "RareParrot", "GuitarTime", "modCheck", "Hmm", "gladge", "Tasty",
    "ResidentSleeper", "FailFish", "NotLikeThis", "SwiftRage", "bleedPurple", "SeemsGood",
    "Prayge", "BibleThump", "BloodTrail", "VoteYea", "RuleFive", "STRONG", "Okayge", "TRUE",
    "Real", "HUH", "Huh", "wat", "NOPE", "SadgeTime", "CLAP", "PauseChamp", "Sussy", "AMOGUS",
)

# Фразы-маркеры реакции/события (латиница и кириллица, регистр не важен).
HYPE_PHRASES: tuple[str, ...] = (
    "no way", "clip it", "clip that", "clipped", "clip this", "lets go", "let's go", "letsgo",
    "what the", "oh my god", "holy shit", "holyyy", "insane", "unbelievable", "impossible",
    "wtf", "omg", "lmao", "lmfao", "lol", "hahaha", "khkhkh", "screaming", "crying",
    "godlike", "cracked", "goated", "sick play", "insane play", "poggers", "feels",
    "жесть", "офигеть", "да ладно", "не может быть", "красава", "красавчик", "погнали",
    "ххаха", "ахаха", "хахаха", "ржу", "ору", "сильно", "пиздец", "ахуеть", "ого",
    "ничего себе", "как так", "ну и ну", "это что", "оpom", "ура",
)

# Прямые просьбы заклипать — самый сильный сигнал.
CLIP_REQUEST_PHRASES: tuple[str, ...] = (
    "clip it", "clip that", "clip this", "clipt it", "clip", "clipped", "заклипай", "клип",
    "клипани", "в клип", "клипни",
)

HYPE_LOWER_PHRASES = tuple(p.lower() for p in HYPE_PHRASES)
CLIP_LOWER_PHRASES = tuple(p.lower() for p in CLIP_REQUEST_PHRASES)
_EMOTE_LOOKUP = {e.lower(): e for e in HYPE_EMOTES}
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{2,}")
_LONG_THRESHOLD = 60  # символов
_CAPS_RE = re.compile(r"[A-ZА-Я]{3,}")


@dataclass
class ChatSignals:
    """По-оконные метрики чата (все массивы одной длины)."""

    bucket: float
    times: np.ndarray
    messages: np.ndarray
    rate: np.ndarray
    hype: np.ndarray
    clip_requests: np.ndarray
    unique: np.ndarray
    spam: np.ndarray
    caps: np.ndarray
    long_ratio: np.ndarray
    emotes: Counter = field(default_factory=Counter)
    hype_words: Counter = field(default_factory=Counter)
    top_messages: list[tuple[str, int]] = field(default_factory=list)
    total_messages: int = 0
    unique_users: int = 0
    duration: float = 0.0

    def window(self, start: float, end: float) -> dict[str, float]:
        """Агрегированные метрики окна [start, end] — для отчёта и промпта LLM."""
        if len(self.times) == 0:
            return {"messages": 0.0, "rate": 0.0, "hype": 0.0, "unique": 0.0, "clip_requests": 0.0}
        mask = (self.times + self.bucket > start) & (self.times < end)
        if not mask.any():
            return {"messages": 0.0, "rate": 0.0, "hype": 0.0, "unique": 0.0, "clip_requests": 0.0}
        return {
            "messages": float(self.messages[mask].sum()),
            "rate": float(self.rate[mask].mean()),
            "peak_rate": float(self.rate[mask].max()),
            "hype": float(self.hype[mask].sum()),
            "unique": float(self.unique[mask].mean()),
            "spam": float(self.spam[mask].mean()),
            "caps": float(self.caps[mask].mean()),
            "clip_requests": float(self.clip_requests[mask].sum()),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "bucket": self.bucket,
            "times": self.times.tolist(),
            "messages": self.messages.tolist(),
            "rate": self.rate.tolist(),
            "hype": self.hype.tolist(),
            "clip_requests": self.clip_requests.tolist(),
            "unique": self.unique.tolist(),
            "spam": self.spam.tolist(),
            "caps": self.caps.tolist(),
            "long_ratio": self.long_ratio.tolist(),
            "emotes": dict(self.emotes.most_common(30)),
            "hype_words": dict(self.hype_words.most_common(30)),
            "top_messages": [list(item) for item in self.top_messages[:30]],
            "total_messages": self.total_messages,
            "unique_users": self.unique_users,
            "duration": self.duration,
        }

def hype_score(text: str) -> tuple[int, int, bool]:
    """(хайп-хиты, эмоут-хиты, есть ли просьба заклипать) для одного сообщения."""
    if not text:
        return 0, 0, False
    lowered = text.lower()
    hits = sum(1 for phrase in HYPE_LOWER_PHRASES if phrase in lowered)
    emote_hits = 0
    for token in _TOKEN_RE.findall(text):
        if token.lower() in _EMOTE_LOOKUP:
            emote_hits += 1
    hype = hits + emote_hits
    if text.count("!") >= 3:
        hype += 1
    if len(_CAPS_RE.findall(text)) >= 2:
        hype += 1
    clip = any(phrase in lowered for phrase in CLIP_LOWER_PHRASES)
    return hype, emote_hits, clip


def load_chat(path: Path | str) -> list[ChatMessage]:
    """Читает chat.jsonl и сортирует по времени."""
    messages = [ChatMessage.from_dict(row) for row in read_jsonl(path)]
    messages.sort(key=lambda m: m.offset)
    return messages


def analyze_chat(
    messages: Sequence[ChatMessage],
    duration: float | None = None,
    bucket: float = 5.0,
) -> ChatSignals:
    """Считает по-оконные метрики чата (быстро даже для сотен тысяч сообщений)."""
    bucket = max(1.0, float(bucket))
    if not messages:
        empty = np.zeros(0)
        return ChatSignals(
            bucket=bucket, times=empty, messages=empty, rate=empty, hype=empty,
            clip_requests=empty, unique=empty, spam=empty, caps=empty, long_ratio=empty,
        )

    offsets = np.asarray([m.offset for m in messages], dtype=np.float64)
    span = float(duration) if duration else float(offsets.max()) + bucket
    buckets = int(np.ceil(span / bucket)) + 1
    index = np.clip((offsets / bucket).astype(np.int64), 0, buckets - 1)

    counts = np.bincount(index, minlength=buckets).astype(np.float64)
    hype = np.zeros(buckets)
    clip_requests = np.zeros(buckets)
    caps = np.zeros(buckets)
    long_ratio = np.zeros(buckets)
    texts_per_bucket: list[list[str]] = [[] for _ in range(buckets)]

    emotes: Counter = Counter()
    hype_words: Counter = Counter()
    user_messages: Counter = Counter()
    message_freq: Counter = Counter()

    for position, message in enumerate(messages):
        slot = int(index[position])
        score, _emote_hits, clip = hype_score(message.text)
        hype[slot] += score
        if clip:
            clip_requests[slot] += 1
        caps[slot] += 1 if _CAPS_RE.search(message.text) else 0
        long_ratio[slot] += 1 if len(message.text) >= _LONG_THRESHOLD else 0
        seen_tokens = set()
        for token in _TOKEN_RE.findall(message.text):
            lowered = token.lower()
            if lowered in _EMOTE_LOOKUP and lowered not in seen_tokens:
                seen_tokens.add(lowered)
                emotes[_EMOTE_LOOKUP[lowered]] += 1
        lowered_text = message.text.lower()
        for phrase in HYPE_LOWER_PHRASES:
            if phrase in lowered_text:
                hype_words[phrase] += 1
        normalized = message.text.strip().lower()[:80]
        if normalized:
            texts_per_bucket[slot].append(normalized)
        if message.login:
            user_messages[message.login] += 1
        stripped = message.text.strip()
        if stripped:
            message_freq[stripped] += 1

    unique_counts = np.zeros(buckets)
    spam = np.zeros(buckets)
    for slot in range(buckets):
        texts = texts_per_bucket[slot]
        if not texts:
            continue
        unique_counts[slot] = len(set(texts))
        if len(texts) >= 4:
            spam[slot] = 1.0 - (len(set(texts)) / len(texts))

    safe_counts = np.maximum(counts, 1.0)
    return ChatSignals(
        bucket=bucket,
        times=(np.arange(buckets, dtype=np.float64) + 0.5) * bucket,
        messages=counts,
        rate=counts / bucket,
        hype=hype / bucket,
        clip_requests=clip_requests,
        unique=unique_counts / bucket,
        spam=spam,
        caps=caps / safe_counts,
        long_ratio=long_ratio / safe_counts,
        emotes=emotes,
        hype_words=hype_words,
        top_messages=message_freq.most_common(50),
        total_messages=len(messages),
        unique_users=len(user_messages),
        duration=max(span, float(offsets.max()) + bucket),
    )


def chat_excerpt(
    messages: Sequence[ChatMessage],
    start: float,
    end: float,
    limit: int = 60,
) -> list[str]:
    """Выжимка чата для окна: приоритет — реакции, эмоуты и просьбы заклинать."""
    scored: list[tuple[float, str]] = []
    for message in messages:
        if not (start <= message.offset <= end):
            continue
        text = message.text.strip()
        if not text:
            continue
        score, _emotes, clip = hype_score(text)
        if clip:
            score += 3.0
        scored.append((score, f"[{int(message.offset - start):>4}s] {text}"))
    scored.sort(key=lambda item: -item[0])
    return [line for _, line in scored[:limit]]

