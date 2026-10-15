"""Конфигурация инструмента: значения по умолчанию, TOML-файл, переменные окружения, CLI."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, get_type_hints

from .modes import mode_name

# Публичный Client-ID веб-плеера Twitch — используется только для чтения публичных данных (VOD/чат).
TWITCH_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"
TWITCH_GQL_URL = "https://gql.twitch.tv/gql"

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OPENAI_URL = "https://api.openai.com/v1"


@dataclass
class Settings:
    """Все настройки пайплайна."""

    # --- общее ---
    output_dir: Path = Path("output")
    config_file: Path | None = None
    verbose: bool = False
    force: bool = False
    limit_seconds: float = 0.0  # 0 = без ограничения (первые N секунд VOD, удобно для тестов)
    # Мод оформления/раскладки: default | toparachevt (см. modes.py).
    mode: str = "default"

    # --- чат ---
    chat_stride: float = 45.0  # шаг сканирования в "тихих" местах, сек
    chat_delay: float = 0.2  # пауза между запросами к GQL, сек
    chat_directions: bool = True  # тратить больше запросов на плотные участки
    chat_gap_audit: bool = True  # искать пропуски после основного прохода
    chat_min_gap: float = 240.0  # разрыв, который считается подозрительным, сек
    chat_heartbeat: float = 30.0  # печатать прогресс каждые N сек прогресса

    # --- аудио / ASR ---
    audio_sample_rate: int = 16000
    whisper_model: str = "medium"
    whisper_device: str = "auto"  # auto|cuda|cpu
    whisper_compute_type: str = "auto"  # auto|float16|int8_float16|int8|float32
    whisper_beam_size: int = 5
    whisper_vad: bool = True
    whisper_batch_size: int = 8
    whisper_threads: int = 0
    asr_backend: str = "auto"  # auto|faster-whisper|openai-whisper
    asr_optional: bool = True  # продолжать анализ без расшифровки, если модель недоступна
    models_dir: Path = Path("models")  # локальные/скачанные веса моделей
    hf_endpoint: str | None = None  # например https://hf-mirror.com — зеркало HuggingFace
    # Язык стрима. По умолчанию "ru" — русские стримы больше не нужно угадывать
    # (автоопределение регулярно путало ru с en). Значение "auto" или пустая строка
    # возвращают автоопределение.
    language: str | None = "ru"
    translate_to: str | None = None  # перевод транскрипта через LLM (например, "ru")

    # --- LLM ---
    llm_provider: str = "auto"  # auto|none|ollama|openai
    llm_model: str = "Qwythos-9B-v2-Heretic:Q4_K_M"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_timeout: float = 240.0
    llm_max_candidates: int = 32  # сколько кандидатов отправлять в LLM
    llm_batch: int = 8

    # --- детект моментов ---
    bucket: float = 5.0  # окно агрегации сигналов, сек
    top_candidates: int = 40
    min_gap: float = 45.0  # минимальное расстояние между клипами, сек
    max_clips: int = 15
    min_moments: int = 0  # гарантированно выдать столько моментов (0 = не добивать)
    clip_length: float = 40.0
    clip_padding: float = 6.0
    min_clip_length: float = 12.0
    max_clip_length: float = 120.0

    # веса сигналов
    w_chat_rate: float = 1.0
    w_chat_hype: float = 0.9
    w_chat_unique: float = 0.35
    w_chat_spam: float = -0.45
    w_audio: float = 0.7
    w_keywords: float = 1.15
    w_laughter: float = 0.5

    # --- нарезка ---
    quality: str = "1080p60"  # best|1080p60|720p|480p|360p|audio
    vertical: bool = False
    burn_subs: bool = False
    precise_cut: bool = False  # точная резка по кадрам (перекодирование секции)
    cut_workers: int = 2
    thumbnail: bool = True

    # --- прочее ---
    cookies_from_browser: str | None = None  # например "chrome" — для приватных VOD


def _sets_from_env() -> dict[str, Any]:
    """OPENAI_API_KEY / TWITCH_* / *_BASE_URL и т.п."""
    values: dict[str, Any] = {}
    if os.getenv("OPENAI_API_KEY"):
        values["llm_api_key"] = os.environ["OPENAI_API_KEY"]
    if os.getenv("OPENAI_BASE_URL"):
        values["llm_base_url"] = os.environ["OPENAI_BASE_URL"]
    if os.getenv("OLLAMA_BASE_URL"):
        values["llm_base_url"] = os.environ["OLLAMA_BASE_URL"]
    if os.getenv("TWITCH_CLIP_OUTPUT"):
        values["output_dir"] = Path(os.environ["TWITCH_CLIP_OUTPUT"])
    if os.getenv("HF_ENDPOINT"):
        values["hf_endpoint"] = os.environ["HF_ENDPOINT"]
    return values


def _coerce(value: Any, type_hint: Any) -> Any:
    """Приводит значения из TOML/CLI к типу поля."""
    if isinstance(type_hint, str):  # неразрешённая аннотация (from __future__ import annotations)
        mapping = {
            "int": int,
            "float": float,
            "bool": bool,
            "str": str,
            "Path": Path,
        }
        type_hint = mapping.get(type_hint.split("|")[0].strip().replace("None", ""), Any)
    origin = getattr(type_hint, "__origin__", None)
    args = getattr(type_hint, "__args__", ())
    if type_hint is Path:
        return Path(value)
    if origin is None and str(type_hint).startswith("Path"):
        return Path(value) if value is not None else None
    if origin is not None and type(None) in args:  # Optional[...]
        inner = [a for a in args if a is not type(None)][0]
        if isinstance(value, str) and value.lower() in {"none", "null", ""}:
            return None
        return _coerce(value, inner)
    if type_hint is bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on", "да"}
    if type_hint is int:
        return int(float(value))
    if type_hint is float:
        return float(value)
    return value


def load_settings(
    config_path: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Собирает настройки: значения по умолчанию <- TOML <- env <- overrides (CLI)."""
    settings = Settings()
    valid = {f.name for f in fields(Settings)}
    try:
        hints = get_type_hints(Settings)
    except Exception:  # noqa: BLE001
        hints = {f.name: f.type for f in fields(Settings)}

    data: dict[str, Any] = {}
    candidate = Path(config_path) if config_path else Path("config.toml")
    if candidate.exists():
        try:
            with candidate.open("rb") as fh:
                raw = tomllib.load(fh)
            for section in ("general", "generals", "chat", "asr", "llm", "detect", "cut", "clip"):
                values = raw.pop(section, None)
                if isinstance(values, dict):
                    data.update(values)
            data.update(raw)
        except tomllib.TOMLDecodeError:
            pass
        settings.config_file = candidate

    merged: dict[str, Any] = {**data, **_sets_from_env(), **(overrides or {})}
    for key, value in merged.items():
        if key not in valid or value is None:
            continue
        try:
            setattr(settings, key, _coerce(value, hints.get(key, Any)))
        except (TypeError, ValueError):
            continue
    if isinstance(settings.output_dir, str):
        settings.output_dir = Path(settings.output_dir)
    # "auto"/"" = автоопределение языка (иначе Whisper получил бы невалидный код "auto").
    if isinstance(settings.language, str) and settings.language.strip().lower() in {"", "auto", "any"}:
        settings.language = None
    # Мод приводим к каноническому имени (принимаем и "топарашевт").
    if isinstance(settings.mode, str):
        settings.mode = mode_name(settings.mode)
    return settings


def settings_as_dict(settings: Settings) -> dict[str, Any]:
    from .utils import to_serializable

    return to_serializable(settings)
