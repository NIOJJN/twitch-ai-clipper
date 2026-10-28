"""LLM-слой: провайдеры (Ollama / OpenAI-совместимые) и прикладные задачи.

Без LLM пайплайн тоже полностью работоспособен (эвристика), но с моделью
появляются осмысленные заголовки, категории и переранжирование по смыслу.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Sequence

from loguru import logger

from .asr import TranscriptSegment
from .config import DEFAULT_OLLAMA_URL, DEFAULT_OPENAI_URL, Settings

CATEGORIES = (
    "clutch", "skill", "funny", "fail", "rage", "reaction", "story", "drama", "hype", "other",
)

RANK_SYSTEM = """Ты — профессиональный редактор Twitch-хайлайтов. Тебе дают фрагменты
стрима: расшифровку речи стримера/игры и чат зрителей. Определи, какие фрагменты
действительно интересны как короткие клипы (яркая реакция, клатч, смешной момент,
провал, драма, сильная эмоция, важное событие).

Отвечай СТРОГО валидным JSON без пояснений и без markdown:
{"moments":[{"index":<номер кандидата>,"score":<0-100>,"title":"<до 60 символов, на языке стрима>","category":"<одна из: clutch|skill|funny|fail|rage|reaction|story|drama|hype|other>","reason":"<1-2 предложения почему это стоит клипать>","start":<секунда начала клипа>,"end":<секунда конца клипа>}]}

ВАЖНО: верни ровно по одному объекту на КАЖДЫЙ входной кандидат, с его index.
Ничего не пропускай и не объединяй: слабые моменты тоже оцени, просто низким
score (мусор/тишина/реклама — ниже 40). Название title — короткое, ёмкое, как
подзаголовок в описании видео ("Любимый цвет", "Пихта хочет в Питер"), а не
начало предложения из расшифровки.
start/end указывай относительно начала VOD (как в данных), длина клипа 15-60 секунд,
начало — за 3-8 секунд до события."""

TRANSLATE_SYSTEM = """Ты — переводчик субтитров. Переведи каждую реплику, сохранив смысл,
разговорный стиль и обращения зрителей. НЕ добавляй пояснения.

Отвечай СТРОГО валидным JSON: {"items":[{"i":<номер>,"t":"<перевод>"}]}."""

SUMMARY_SYSTEM = "Ты кратко резюмируешь стримы для отчёта."


class LLMError(RuntimeError):
    """Ошибка обращения к LLM."""


def extract_json(text: str) -> Any:
    """Достаёт первый JSON-объект/массив из ответа модели (в т.ч. из ```json-блока)."""
    if not text:
        raise LLMError("пустой ответ модели")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    raise LLMError(f"не удалось разобрать JSON из ответа: {cleaned[:300]}")


class LLMProvider:
    """Базовый провайдер."""

    name = "base"
    model = ""

    def available(self) -> bool:
        return False

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        raise NotImplementedError

    def json_chat(self, system: str, user: str, temperature: float = 0.2) -> Any:
        return extract_json(self.chat(system, user, temperature, json_mode=True))


class NullProvider(LLMProvider):
    """Заглушка: LLM отключён, работает чистая эвристика."""

    name = "none"

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        raise LLMError("LLM отключён (--llm none)")


class OllamaProvider(LLMProvider):
    """Локальная модель через Ollama (http://127.0.0.1:11434)."""

    name = "ollama"

    def __init__(self, model: str, base_url: str | None = None, timeout: float = 240.0) -> None:
        self.model = model
        self.base_url = (base_url or DEFAULT_OLLAMA_URL).rstrip("/")
        self.timeout = timeout

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3) as response:
                data = json.loads(response.read().decode("utf-8"))
            return bool(data.get("models"))
        except Exception:  # noqa: BLE001
            return False

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": 8192,
                "num_predict": 2048,
            },
        }
        if json_mode:
            payload["format"] = "json"
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise LLMError(f"Ollama недоступен: {exc}") from exc
        return ((data.get("message") or {}).get("content") or "").strip()


class OpenAICompatProvider(LLMProvider):
    """Любой OpenAI-совместимый API: OpenAI, OpenRouter, LM Studio, vLLM, llama.cpp."""

    name = "openai"

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 240.0,
    ) -> None:
        self.model = model
        self.base_url = (base_url or DEFAULT_OPENAI_URL).rstrip("/")
        self.api_key = api_key or ""
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key) or "localhost" in self.base_url or "127.0.0.1" in self.base_url

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:  # noqa: BLE001
                pass
            raise LLMError(f"LLM HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"LLM недоступен ({self.base_url}): {exc}") from exc
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"пустой ответ LLM: {str(data)[:200]}")
        return (choices[0].get("message") or {}).get("content", "").strip()


def get_provider(settings: Settings) -> LLMProvider:
    """Выбирает провайдера: auto -> Ollama, затем OpenAI-совместимый API, иначе null."""
    choice = (settings.llm_provider or "auto").lower()
    if choice in {"none", "off", "no"}:
        logger.info("LLM: отключён, работаем на эвристике")
        return NullProvider()

    if choice in {"ollama", "auto"}:
        provider = OllamaProvider(settings.llm_model, settings.llm_base_url, settings.llm_timeout)
        if provider.available():
            logger.info("LLM: Ollama, модель {}", provider.model)
            return provider
        if choice == "ollama":
            logger.warning("LLM: Ollama недоступен ({})", provider.base_url)

    remote_hint = bool(settings.llm_api_key) or bool(
        settings.llm_base_url and "11434" not in (settings.llm_base_url or "")
    )
    if choice == "openai" or (choice == "auto" and remote_hint):
        provider = OpenAICompatProvider(
            settings.llm_model, settings.llm_base_url, settings.llm_api_key, settings.llm_timeout
        )
        if provider.available():
            logger.info(
                "LLM: OpenAI-совместимый API {}, модель {}", provider.base_url, provider.model
            )
            return provider
        logger.warning("LLM: API недоступен ({})", provider.base_url)

    logger.info("LLM: не найден — работаем на эвристике (заголовки и рейтинг считаем сами)")
    return NullProvider()


@dataclass
class LLMRating:
    """Оценка кандидата моделью."""

    index: int
    score: float = 0.0
    title: str = ""
    category: str = "other"
    reason: str = ""
    start: float | None = None
    end: float | None = None


def rank_moments(
    provider: LLMProvider,
    items: Sequence[dict[str, Any]],
    vod_title: str = "",
    language: str = "",
    batch: int = 8,
    chat_limit: int = 40,
) -> dict[int, LLMRating]:
    """Просит LLM оценить и назвать кандидатов. Возвращает {index: LLMRating}."""
    ratings: dict[int, LLMRating] = {}
    if not items or isinstance(provider, NullProvider):
        return ratings

    def ask(chunk: Sequence[dict[str, Any]], number: int) -> int:
        """Один батч -> сколько новых оценок получили."""
        payload = {
            "vod_title": vod_title,
            "stream_language": language,
            "candidates": [
                {
                    "index": item["index"],
                    "start": round(float(item["start"]), 1),
                    "end": round(float(item["end"]), 1),
                    "signals": item.get("signals", {}),
                    "transcript": (item.get("transcript") or "")[:1800],
                    "chat": (item.get("chat") or [])[:chat_limit],
                }
                for item in chunk
            ],
        }
        user = "Фрагменты стрима (JSON):\n" + json.dumps(payload, ensure_ascii=False)
        data: Any = None
        for attempt in (1, 2):
            try:
                data = provider.json_chat(RANK_SYSTEM, user)
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM: батч {} попытка {} не удалась: {}", number, attempt, exc)
        if data is None:
            return 0
        moments = data.get("moments") if isinstance(data, dict) else data
        if not isinstance(moments, list):
            logger.warning("LLM: неожиданный формат ответа батча {}", number)
            return 0
        added = 0
        for moment in moments:
            if not isinstance(moment, dict):
                continue
            try:
                index = int(moment.get("index"))
            except (TypeError, ValueError):
                continue
            category = str(moment.get("category") or "other").lower()
            ratings[index] = LLMRating(
                index=index,
                score=float(moment.get("score") or 0.0),
                title=str(moment.get("title") or "").strip(),
                category=category if category in CATEGORIES else "other",
                reason=str(moment.get("reason") or "").strip(),
                start=_maybe_float(moment.get("start")),
                end=_maybe_float(moment.get("end")),
            )
            added += 1
        return added

    chunks = [items[i : i + batch] for i in range(0, len(items), batch)]
    for number, chunk in enumerate(chunks, start=1):
        ask(chunk, number)
        logger.info("LLM: оценено кандидатов {}/{}", len(ratings), len(items))

    # Модель иногда молча выбрасывает часть кандидатов. Для оглавления нужна
    # каждая глава, поэтому неоценённые спрашиваем отдельно, по одному.
    missing = [item for item in items if item["index"] not in ratings]
    if missing:
        logger.info("LLM: добиваем пропущенные кандидаты: {}", len(missing))
        for offset, item in enumerate(missing, start=1):
            ask([item], len(chunks) + offset)
        logger.info("LLM: оценено кандидатов {}/{}", len(ratings), len(items))
    return ratings


def _maybe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def translate_segments(
    provider: LLMProvider,
    segments: list[TranscriptSegment],
    target: str,
    batch: int = 20,
    progress: Any | None = None,
) -> int:
    """Переводит реплики транскрипта через LLM (заполняет ``segment.translation``)."""
    if not segments or isinstance(provider, NullProvider) or not target:
        return 0
    translated = 0
    indices = list(range(len(segments)))
    chunks = [indices[i : i + batch] for i in range(0, len(indices), batch)]
    for number, chunk in enumerate(chunks, start=1):
        payload = {
            "target_language": target,
            "items": [{"i": i, "t": segments[i].text[:600]} for i in chunk],
        }
        try:
            data = provider.json_chat(
                TRANSLATE_SYSTEM, json.dumps(payload, ensure_ascii=False)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Перевод: батч {} не удался: {}", number, exc)
            continue
        items = data.get("items") if isinstance(data, dict) else data
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("i"))
            except (TypeError, ValueError):
                continue
            text = str(item.get("t") or "").strip()
            if 0 <= index < len(segments) and text:
                segments[index].translation = text
                translated += 1
        if progress is not None:
            progress(number, len(chunks))
        logger.info("Перевод: {}/{} батчей", number, len(chunks))
    return translated


def summarize_stream(
    provider: LLMProvider,
    vod_title: str,
    duration_text: str,
    highlights: Sequence[dict[str, Any]],
    chat_stats: dict[str, Any] | None = None,
) -> str:
    """Короткое резюме стрима для отчёта (пустая строка, если LLM нет)."""
    if isinstance(provider, NullProvider):
        return ""
    lines = [f"Стрим: {vod_title} ({duration_text})"]
    if chat_stats:
        lines.append(
            f"Чат: {chat_stats.get('total_messages', 0)} сообщений, "
            f"{chat_stats.get('unique_users', 0)} уникальных зрителей"
        )
    for item in highlights[:15]:
        lines.append(
            f"- {item.get('time_text', '')} [{item.get('category', '')}] {item.get('title', '')}"
        )
    user = "\n".join(lines) + "\n\nНапиши краткое резюме стрима (3-5 предложений)."
    try:
        return provider.chat(SUMMARY_SYSTEM, user).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM: резюме не сделано: {}", exc)
        return ""