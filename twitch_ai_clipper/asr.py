"""Транскрипция аудио через faster-whisper (CUDA, если доступна) и экспорт субтитров."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from .config import Settings
from .utils import (
    enable_cuda_dll_dirs,
    ensure_dir,
    fmt_ts,
    read_jsonl,
    write_json,
    write_jsonl,
)

# HuggingFace "xet" транспорт в части сетей уходит в вечное ожидание — по умолчанию
# отключаем его (значение можно переопределить переменной окружения).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Нужен до загрузки модели: CTranslate2 ищет cublas64_12.dll, а нередко в системе есть
# только CUDA 11 из torch. Подключаем bin-каталоги pip-пакетов nvidia/*.
enable_cuda_dll_dirs()


DEVICE_CANDIDATES: dict[str, list[tuple[str, str]]] = {
    "auto": [("cuda", "float16"), ("cuda", "int8_float16"), ("cpu", "int8"), ("cpu", "float32")],
    "cuda": [("cuda", "float16"), ("cuda", "int8_float16"), ("cuda", "float32")],
    "cpu": [("cpu", "int8"), ("cpu", "float32")],
}


@dataclass
class TranscriptSegment:
    """Реплика расшифровки с таймкодами."""

    start: float
    end: float
    text: str
    translation: str = ""

    @property
    def display(self) -> str:
        return self.translation or self.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "text": self.text,
            "translation": self.translation,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "TranscriptSegment":
        return cls(
            start=float(row.get("start", 0.0)),
            end=float(row.get("end", 0.0)),
            text=row.get("text", ""),
            translation=row.get("translation", "") or "",
        )


@dataclass
class TranscriptResult:
    """Результат расшифровки целиком."""

    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str = ""
    language_probability: float = 0.0
    duration: float = 0.0
    model: str = ""
    device: str = ""
    compute_type: str = ""
    elapsed: float = 0.0
    audio_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        words = sum(len(s.text.split()) for s in self.segments)
        return {
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "model": self.model,
            "device": self.device,
            "compute_type": self.compute_type,
            "elapsed": self.elapsed,
            "audio_path": self.audio_path,
            "segments": len(self.segments),
            "words": words,
        }


def load_model(settings: Settings) -> tuple[Any, dict[str, str]]:
    """Загружает WhisperModel, перебирая варианты device/compute_type до рабочего."""
    from faster_whisper import WhisperModel

    if settings.hf_endpoint:
        os.environ["HF_ENDPOINT"] = settings.hf_endpoint

    candidates = list(DEVICE_CANDIDATES.get(settings.whisper_device, DEVICE_CANDIDATES["auto"]))
    if settings.whisper_compute_type != "auto":
        candidates.sort(
            key=lambda pair: 0 if pair[1] == settings.whisper_compute_type else 1
        )
    errors: list[str] = []
    for device, compute_type in candidates:
        try:
            started = time.time()
            kwargs: dict[str, Any] = {"device": device, "compute_type": compute_type}
            if settings.whisper_threads:
                kwargs["cpu_threads"] = int(settings.whisper_threads)
            model = WhisperModel(settings.whisper_model, **kwargs)
            logger.info(
                "Whisper '{}' загружен: device={} compute={} ({:.1f} с)",
                settings.whisper_model, device, compute_type, time.time() - started,
            )
            return model, {"device": device, "compute_type": compute_type}
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{device}/{compute_type}: {type(exc).__name__}: {str(exc)[:160]}")
            logger.warning("Whisper {}/{} недоступен -> {}", device, compute_type, str(exc)[:160])
    raise RuntimeError("Не удалось загрузить модель Whisper:\n" + "\n".join(errors))


def transcribe_faster_whisper(
    wav_path: Path | str,
    settings: Settings,
    model: Any | None = None,
    progress: Callable[[float, float], None] | None = None,
) -> TranscriptResult:
    """Расшифровывает WAV-файл целиком; ``progress(done_sec, total_sec)`` — прогресс."""
    wav_path = Path(wav_path)
    model_info = {
        "device": getattr(model, "device", settings.whisper_device),
        "compute_type": getattr(model, "compute_type", settings.whisper_compute_type),
    }
    started = time.time()
    options: dict[str, Any] = {
        "language": settings.language or None,
        "task": "transcribe",
        "beam_size": max(1, int(settings.whisper_beam_size)),
        "vad_filter": bool(settings.whisper_vad),
        "vad_parameters": {"min_silence_duration_ms": 500, "speech_pad_ms": 200},
        "condition_on_previous_text": False,
        "word_timestamps": False,
        "without_timestamps": False,
    }
    runner: Any = model
    batch_size = max(0, int(settings.whisper_batch_size or 0))
    if batch_size > 1:
        # Батчевое декодирование CTranslate2: на многочасовых стримах заметно быстрее
        # последовательного. Если что-то пойдёт не так (старая версия/OOM) — откат ниже.
        try:
            from faster_whisper import BatchedInferencePipeline

            runner = BatchedInferencePipeline(model=model)
            options["batch_size"] = batch_size
            logger.info(
                "Батчевая расшифровка: batch_size={}, язык {}",
                batch_size, settings.language or "авто",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Батчевая расшифровка недоступна ({}), работаем по одному", exc)
            runner = model
    def collect(segments_iter: Any, info: Any) -> TranscriptResult:
        """Разбирает ленивый итератор faster-whisper в TranscriptResult."""
        total = float(getattr(info, "duration", 0.0) or 0.0)
        result = TranscriptResult(
            language=getattr(info, "language", "") or "",
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            duration=total,
            model=settings.whisper_model,
            device=str(model_info.get("device", "")),
            compute_type=str(model_info.get("compute_type", "")),
            audio_path=str(wav_path),
        )
        next_report = 0.0
        for segment in segments_iter:
            text = (segment.text or "").strip()
            if not text:
                continue
            result.segments.append(
                TranscriptSegment(start=float(segment.start), end=float(segment.end), text=text)
            )
            if progress is not None:
                progress(float(segment.end), total)
            if total and float(segment.end) >= next_report:
                logger.info(
                    "Расшифровка: {:5.1f}% ({})",
                    min(100.0, float(segment.end) / total * 100),
                    fmt_ts(segment.end),
                )
                next_report = float(segment.end) + max(60.0, total * 0.02)
        result.elapsed = time.time() - started
        return result

    try:
        result = collect(*runner.transcribe(str(wav_path), **options))
    except RuntimeError as exc:
        if runner is model:
            raise
        # Не хватило видеопамяти на пакетном декодировании — продолжаем без батчей.
        logger.warning(
            "Батчевая расшифровка упала ({}), повторяем последовательно", str(exc)[:200]
        )
        options.pop("batch_size", None)
        result = collect(*model.transcribe(str(wav_path), **options))

    logger.info(
        "Расшифровка готова: {} реплик, язык {} ({:.2f}); {:.1f} мин аудио за {:.1f} мин",
        len(result.segments),
        result.language or "?",
        result.language_probability,
        result.duration / 60 if result.duration else 0.0,
        result.elapsed / 60,
    )
    return result


def _srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total_millis = int(round(seconds * 1000))  # округение единое, иначе возможны "...,1000"
    seconds, millis = divmod(total_millis, 1000)
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class AsrError(RuntimeError):
    """Не удалось загрузить/запустить модель распознавания речи."""


ASR_HINT = (
    "Модель Whisper не загрузилась. Варианты решения:\n"
    "  1) fastest: установите faster-whisper и дайте доступ к huggingface.co "
    "(или задайте HF_ENDPOINT=https://hf-mirror.com);\n"
    "  2) fallback: pip install openai-whisper — он качает веса с openaipublic.azureedge.net; "
    "запустите с --asr-backend openai-whisper;\n"
    "  3) положите готовую CT2-модель локально и укажите путь: "
    "--whisper-model D:\\models\\faster-whisper-medium;\n"
    "  4) чтобы просто пропустить распознавание: --asr-optional (анализ пойдёт по чату и звуку)."
)


def probe_audio_duration(wav_path: Path | str) -> float:
    """Длительность WAV по заголовку (без декодирования — через soundfile)."""
    try:
        import soundfile as sf

        return float(sf.info(str(wav_path)).duration)
    except Exception:  # noqa: BLE001
        return 0.0


def transcribe_openai_whisper(
    wav_path: Path | str,
    settings: Settings,
    model: Any | None = None,
    progress: Callable[[float, float], None] | None = None,
) -> TranscriptResult:
    """Запасной бэкенд: openai-whisper (веса тянутся с openaipublic.azureedge.net)."""
    import whisper  # установка: pip install openai-whisper

    device = "cuda" if settings.whisper_device in {"auto", "cuda"} and _cuda_available() else "cpu"
    if settings.whisper_device == "cpu":
        device = "cpu"
    download_root = Path(settings.models_dir)
    ensure_dir(download_root)
    if model is None:
        logger.info("Загружаем openai-whisper '{}' ({}) ...", settings.whisper_model, device)
        started = time.time()
        model = whisper.load_model(
            settings.whisper_model, device=device, download_root=str(download_root)
        )
        logger.info("openai-whisper загружен за {:.1f} с", time.time() - started)

    started = time.time()
    options: dict[str, Any] = {
        "language": settings.language or None,
        "verbose": None,
        "fp16": device == "cuda",
        "condition_on_previous_text": False,
        "beam_size": max(1, int(settings.whisper_beam_size)),
    }
    total_audio = probe_audio_duration(wav_path)
    logger.info(
        "Расшифровка запущена: {} ({} мин аудио на {}). Это надолго — можно следить по логу.",
        Path(wav_path).name, round(total_audio / 60), device,
    )
    raw = model.transcribe(str(wav_path), **options)
    segments = [
        TranscriptSegment(
            start=float(item["start"]), end=float(item["end"]), text=str(item["text"]).strip()
        )
        for item in raw.get("segments", [])
        if str(item.get("text", "")).strip()
    ]
    duration = float(segments[-1].end) if segments else 0.0
    result = TranscriptResult(
        segments=segments,
        language=str(raw.get("language") or ""),
        language_probability=0.0,
        duration=duration,
        model=f"openai-whisper:{settings.whisper_model}",
        device=device,
        compute_type="fp16" if device == "cuda" else "fp32",
        elapsed=time.time() - started,
        audio_path=str(wav_path),
    )
    logger.info(
        "Расшифровка (openai-whisper): {} реплик, язык {}, {:.1f} мин",
        len(segments), result.language or "?", result.elapsed / 60,
    )
    return result


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def load_backend(settings: Settings) -> tuple[str, Any, dict[str, str]]:
    """Готовит бэкенд распознавания: faster-whisper, при неудаче — openai-whisper."""
    backend = (settings.asr_backend or "auto").lower()
    errors: list[str] = []

    if backend in {"auto", "faster", "faster-whisper"}:
        try:
            model, info = load_model(settings)
            return "faster-whisper", model, info
        except Exception as exc:  # noqa: BLE001
            errors.append(f"faster-whisper: {type(exc).__name__}: {str(exc)[:200]}")
            logger.warning("faster-whisper недоступен: {}", str(exc)[:200])
            if backend in {"faster", "faster-whisper"}:
                raise AsrError(ASR_HINT + "\n\n" + "\n".join(errors)) from exc

    if backend in {"auto", "openai", "openai-whisper", "whisper"}:
        try:
            import importlib

            importlib.import_module("whisper")
            logger.info("Переключаемся на бэкенд openai-whisper")
            return "openai-whisper", None, {"device": "auto", "compute_type": "auto"}
        except Exception as exc:  # noqa: BLE001
            errors.append(f"openai-whisper: {type(exc).__name__}: {str(exc)[:120]}")

    raise AsrError(ASR_HINT + "\n\n" + "\n".join(errors))


def transcribe(
    wav_path: Path | str,
    settings: Settings,
    model: Any | None = None,
    progress: Callable[[float, float], None] | None = None,
    backend: str | None = None,
) -> TranscriptResult:
    """Расшифровка с автоматическим выбором рабочего бэкенда."""
    if backend == "openai-whisper" or (backend is None and settings.asr_backend == "openai-whisper"):
        return transcribe_openai_whisper(wav_path, settings, model=model, progress=progress)
    if backend == "faster-whisper" or (
        backend is None and settings.asr_backend in {"faster", "faster-whisper"}
    ):
        return transcribe_faster_whisper(wav_path, settings, model=model, progress=progress)
    kind, loaded, info = load_backend(settings)
    if kind == "openai-whisper":
        return transcribe_openai_whisper(wav_path, settings, model=loaded, progress=progress)
    return transcribe_faster_whisper(wav_path, settings, model=loaded, progress=progress)


def write_transcript_files(
    result: TranscriptResult,
    out_dir: Path | str,
    prefix: str = "transcript",
    use_translation: bool = True,
) -> dict[str, str]:
    """Пишет transcript.jsonl / .txt / .srt / .meta.json и возвращает пути."""
    out_dir = ensure_dir(out_dir)
    jsonl_path = out_dir / f"{prefix}.jsonl"
    write_jsonl(jsonl_path, [seg.to_dict() for seg in result.segments])
    write_json(out_dir / f"{prefix}.meta.json", result.to_dict())

    text_lines = []
    srt_lines = []
    for index, segment in enumerate(result.segments, start=1):
        text = segment.display if use_translation else segment.text
        text_lines.append(f"[{fmt_ts(segment.start)}] {segment.text}")
        srt_lines.append(str(index))
        srt_lines.append(f"{_srt_time(segment.start)} --> {_srt_time(segment.end)}")
        srt_lines.append(text)
        srt_lines.append("")
    txt_path = out_dir / f"{prefix}.txt"
    txt_path.write_text("\n".join(text_lines), encoding="utf-8")
    srt_path = out_dir / f"{prefix}.srt"
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
    return {"jsonl": str(jsonl_path), "txt": str(txt_path), "srt": str(srt_path)}


def load_transcript(path: Path | str) -> list[TranscriptSegment]:
    """Читает transcript.jsonl."""
    return [TranscriptSegment.from_dict(row) for row in read_jsonl(path)]


def segments_in_range(
    segments: list[TranscriptSegment], start: float, end: float
) -> list[TranscriptSegment]:
    """Реплики, пересекающиеся с окном [start, end]."""
    return [seg for seg in segments if seg.end >= start and seg.start <= end]


def transcript_excerpt(
    segments: list[TranscriptSegment],
    start: float,
    end: float,
    max_chars: int = 2500,
    use_translation: bool = True,
) -> str:
    """Текст окна с относительными таймкодами — компактно для промпта LLM."""
    parts: list[str] = []
    for segment in segments_in_range(segments, start, end):
        text = segment.display if use_translation else segment.text
        if not text:
            continue
        parts.append(f"[{int(max(0.0, segment.start - start)):>4}s] {text}")
        if sum(len(p) for p in parts) > max_chars:
            break
    return "\n".join(parts)


def write_clip_srt(
    segments: list[TranscriptSegment],
    path: Path | str,
    clip_start: float,
    clip_end: float,
    use_translation: bool = True,
) -> Path:
    """SRT для клипа: таймкоды сдвинуты к нулю клипа."""
    lines: list[str] = []
    index = 0
    for segment in segments_in_range(segments, clip_start, clip_end):
        text = (segment.display if use_translation else segment.text).strip()
        if not text:
            continue
        index += 1
        start = max(0.0, segment.start - clip_start)
        end = max(start + 0.4, segment.end - clip_start)
        lines += [str(index), f"{_srt_time(start)} --> {_srt_time(end)}", text, ""]
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

