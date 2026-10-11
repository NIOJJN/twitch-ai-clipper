"""Пайплайн: VOD -> чат -> аудио -> расшифровка -> сигналы -> моменты -> клипы -> отчёт.

Каждый шаг идемпотентен: результаты кешируются в ``output/<vod_id>/``, повторный
запуск не перекачивает и не перерасшифровывает данные (если не указан ``--force``).
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from loguru import logger

from .asr import (
    TranscriptResult,
    TranscriptSegment,
    load_backend,
    load_transcript,
    transcribe,
    write_clip_srt,
    write_transcript_files,
)
from .audio import audio_dynamics, audio_rms, download_audio, probe_duration, to_wav
from .chat import ChatMessage, analyze_chat, load_chat
from .clipper import burn_subtitles, clip_output_path, cut_clip, cut_clip_local, make_thumbnail, make_vertical
from .config import Settings
from .detect import Moment, build_moments
from .features import build_signals, pick_candidates
from .llm import get_provider, summarize_stream, translate_segments
from .report import write_reports
from .twitch_api import TwitchClient, VodInfo
from .utils import ensure_dir, fmt_duration, fmt_ts, is_local_video, read_json, slugify, write_json


@dataclass
class Workspace:
    """Рабочая папка одного VOD."""

    vod: VodInfo
    root: Path
    settings: Settings
    client: TwitchClient = field(repr=False)
    local_path: Path | None = None  # путь к локальному видеофайлу (если источник — файл)

    @property
    def work_dir(self) -> Path:
        return ensure_dir(self.root / "work")

    @property
    def clips_dir(self) -> Path:
        return ensure_dir(self.root / "clips")

    @property
    def chat_path(self) -> Path:
        return self.root / "chat.jsonl"

    @property
    def wav_path(self) -> Path:
        return self.work_dir / "audio16k.wav"

    @property
    def transcript_path(self) -> Path:
        return self.root / "transcript.jsonl"

    @property
    def signals_path(self) -> Path:
        return self.root / "signals.json"

    @property
    def candidates_path(self) -> Path:
        return self.root / "candidates.json"

    @property
    def moments_path(self) -> Path:
        return self.root / "moments.json"


def prepare(target: str, settings: Settings) -> Workspace:
    """Разбирает цель (локальный файл/ссылка/ID/канал), тянет метаданные и создаёт папки."""
    local = is_local_video(target)
    if local is not None:
        return _prepare_local(local, settings)

    client = TwitchClient(delay=settings.chat_delay)
    video_id = client.resolve_vod_id(target)
    vod = client.get_vod(video_id)
    if settings.limit_seconds:
        vod.duration = min(vod.duration, float(settings.limit_seconds))
    root = ensure_dir(Path(settings.output_dir) / vod.id)
    write_json(root / "metadata.json", vod.to_dict())
    logger.info(
        "VOD {} | {} | {} | {} ({})",
        vod.id, vod.channel, fmt_duration(vod.duration), vod.title[:60], vod.game,
    )
    return Workspace(vod=vod, root=root, settings=settings, client=client)


def _prepare_local(path: Path, settings: Settings) -> Workspace:
    """Цель — локальный видеофайл: метаданные из ffprobe, без Twitch и сети."""
    duration = probe_duration(path)
    if duration <= 0:
        raise ValueError(f"Не удалось определить длительность файла: {path} (это точно видео?)")
    vod = VodInfo(
        id=slugify(path.stem, max_len=48, fallback="local"),
        title=path.stem,
        duration=duration,
        channel="",
        channel_login="",
        url=str(path),
        local=True,
    )
    if settings.limit_seconds:
        vod.duration = min(vod.duration, float(settings.limit_seconds))
    root = ensure_dir(Path(settings.output_dir) / vod.id)
    write_json(root / "metadata.json", vod.to_dict())
    logger.info(
        "Локальное видео {} | {} | {}",
        vod.id, fmt_duration(duration), path,
    )
    return Workspace(
        vod=vod,
        root=root,
        settings=settings,
        client=TwitchClient(delay=0.0),
        local_path=path,
    )


# ------------------------------------------------------------------ шаги
def step_chat(ws: Workspace) -> list[ChatMessage]:
    """Шаг 1: чат VOD."""
    settings = ws.settings
    if ws.local_path is not None:
        logger.info("Локальное видео — чат недоступен, детект опирается на звук и речь.")
        return []
    cached = ws.chat_path.exists() and not settings.force
    if not cached:
        ws.client.download_chat(
            ws.vod.id,
            ws.chat_path,
            duration=ws.vod.duration or None,
            stride=settings.chat_stride,
            resume=True,
            gap_audit=settings.chat_gap_audit,
            min_gap=settings.chat_min_gap,
            heartbeat=settings.chat_heartbeat,
        )
    messages = load_chat(ws.chat_path)
    if not messages:
        logger.warning("Чат пуст — детект будет опираться только на звук и речь.")
    else:
        logger.info("Чат: {} сообщений от {} зрителей", len(messages), len({m.login for m in messages}))
    return messages


def step_audio(ws: Workspace) -> Path:
    """Шаг 2: аудиодорожка VOD -> WAV 16 kHz mono (для Whisper и энергии звука)."""
    settings = ws.settings
    if ws.wav_path.exists() and not settings.force:
        logger.info("Аудио уже подготовлено: {}", ws.wav_path.name)
        return ws.wav_path
    if ws.local_path is not None:
        # Локальный файл уже на диске — yt-dlp не нужен, извлекаем дорожку сразу.
        return to_wav(
            ws.local_path,
            ws.wav_path,
            sample_rate=settings.audio_sample_rate,
            start=None,
            duration=settings.limit_seconds or None,
            force=settings.force,
        )
    raw = download_audio(
        ws.vod.url,
        ws.work_dir,
        limit_seconds=settings.limit_seconds,
        cookies_from_browser=settings.cookies_from_browser,
        expected_duration=float(ws.vod.duration or 0.0),
    )
    # При --limit-seconds режем и уже скачанный полный файл, иначе WAV получится
    # на всю длину стрима (для 5-часового VOD это лишние гигабайты и часы распознавания).
    return to_wav(
        raw,
        ws.wav_path,
        sample_rate=settings.audio_sample_rate,
        start=None,
        duration=settings.limit_seconds or None,
        force=settings.force,
    )


def step_transcribe(ws: Workspace) -> TranscriptResult:
    """Шаг 3: расшифровка (faster-whisper) + опциональный перевод через LLM."""
    settings = ws.settings
    if ws.transcript_path.exists() and not settings.force:
        segments = load_transcript(ws.transcript_path)
        meta = read_json(ws.root / "transcript.meta.json", {}) or {}
        logger.info("Расшифровка найдена в кеше: {} реплик", len(segments))
        return TranscriptResult(
            segments=segments,
            language=meta.get("language", ""),
            language_probability=float(meta.get("language_probability", 0.0) or 0.0),
            duration=float(meta.get("duration", 0.0) or 0.0),
            model=str(meta.get("model", settings.whisper_model)),
            device=str(meta.get("device", "")),
            compute_type=str(meta.get("compute_type", "")),
            elapsed=float(meta.get("elapsed", 0.0) or 0.0),
            audio_path=str(meta.get("audio_path", ws.wav_path)),
        )

    wav = step_audio(ws)
    try:
        kind, model, model_info = load_backend(settings)
        result = transcribe(wav, settings, model=model, backend=kind)
        result.device = str(model_info.get("device", result.device) or result.device)
        result.compute_type = str(model_info.get("compute_type", result.compute_type) or result.compute_type)
    except Exception as exc:  # noqa: BLE001
        if not settings.asr_optional:
            raise
        logger.warning("Распознавание речи пропущено: {}", exc)
        result = TranscriptResult(
            segments=[],
            duration=probe_duration(wav),
            model=settings.whisper_model,
            audio_path=str(wav),
        )

    if settings.translate_to and result.segments:
        provider = get_provider(settings)
        translated = translate_segments(provider, result.segments, settings.translate_to)
        logger.info("Перевод на '{}': {}/{} реплик", settings.translate_to, translated, len(result.segments))

    write_transcript_files(result, ws.root, prefix="transcript")
    return result


def step_signals(
    ws: Workspace,
    messages: Sequence[ChatMessage],
    segments: Sequence[TranscriptSegment],
) -> Any:
    """Шаг 4: слияние чата, звука и текста в карту интересности."""
    settings = ws.settings
    chat = analyze_chat(messages, duration=ws.vod.duration or None, bucket=settings.bucket)
    rms_db = np.zeros(0)
    dynamics = np.zeros(0)
    if ws.wav_path.exists():
        rms_db = audio_rms(ws.wav_path, hop=1.0)
        dynamics = audio_dynamics(rms_db, hop=1.0)
    table = build_signals(
        chat,
        settings,
        rms_db=rms_db,
        dynamics=dynamics,
        segments=segments,
        duration=ws.vod.duration or None,
    )
    write_json(
        ws.signals_path,
        {
            "bucket": table.bucket,
            "times": table.times.tolist(),
            "score": table.score.tolist(),
            "components": {k: v.tolist() for k, v in table.components.items()},
            "chat": chat.to_dict(),
        },
    )
    logger.info(
        "Сигналы посчитаны: {} окон по {} с, максимум скора {:.1f}",
        len(table.times), table.bucket, float(table.score.max()) if table.score.size else 0.0,
    )
    return chat, table


def _warn_transcript_coverage(ws: Workspace, segments: Sequence[TranscriptSegment]) -> None:
    """Проверяет, что расшифровка покрывает весь VOD, а не только его начало.

    Без этого моменты за пределами транскрипта получают заголовки из чата
    и «слепые» текстовые сигналы — выглядит так, будто программа смотрит только чат.
    """
    total = float(ws.vod.duration or 0.0)
    if total <= 0:
        return
    if not segments:
        logger.warning(
            "Расшифровки нет ({}): моменты и заголовки будут строиться только по чату и звуку. "
            "Выполните 'transcribe' для анализа речи.",
            ws.transcript_path.name,
        )
        return
    covered = max(segment.end for segment in segments)
    ratio = covered / total
    if ratio < 0.9:
        logger.warning(
            "ВНИМАНИЕ: расшифровка покрывает только {}% стрима ({} с из {} с). "
            "Моменты за пределами {} будут без текста речи — их заголовки придётся брать из чата. "
            "Перезапустите с --force (без --limit-seconds), чтобы расшифровать весь VOD.",
            round(ratio * 100), fmt_ts(covered), fmt_ts(total), fmt_ts(covered),
        )


def step_detect(ws: Workspace) -> tuple[list[Moment], Any]:
    """Шаг 5: кандидаты -> LLM/эвристика -> финальные моменты."""
    settings = ws.settings
    messages = load_chat(ws.chat_path)
    segments = load_transcript(ws.transcript_path) if ws.transcript_path.exists() else []
    _warn_transcript_coverage(ws, segments)
    chat, table = step_signals(ws, messages, segments)
    candidates = pick_candidates(table, settings)
    write_json(ws.candidates_path, [candidate.to_dict() for candidate in candidates])
    logger.info("Кандидатов: {}", len(candidates))
    if not candidates:
        logger.warning("Кандидаты не найдены — проверьте, что чат и аудио скачаны.")
        return [], chat
    provider = get_provider(settings)
    moments = build_moments(
        candidates,
        messages,
        segments,
        settings,
        provider=provider,
        vod_title=ws.vod.title,
        language=settings.language or "",
        duration=ws.vod.duration or None,
    )
    summary = summarize_stream(
        provider,
        ws.vod.title,
        fmt_duration(ws.vod.duration),
        [moment.to_dict(settings.mode) for moment in moments],
        chat.to_dict(),
    )
    write_json(ws.moments_path, [moment.to_dict(settings.mode) for moment in moments])
    if summary:
        (ws.root / "ai_summary.txt").write_text(summary, encoding="utf-8")
    logger.info("Моментов после отбора: {}", len(moments))
    if settings.min_moments and len(moments) < settings.min_moments:
        logger.info(
            "Подсказка: чтобы получить больше моментов, уменьшите --min-gap (сейчас {:.0f} с) "
            "и --min-clip-length (сейчас {:.0f} с).",
            settings.min_gap, settings.min_clip_length,
        )
    return moments, chat


def step_clips(ws: Workspace, moments: Sequence[Moment]) -> list[Moment]:
    """Шаг 6: нарезка клипов (yt-dlp по секциям + ffmpeg-обработка)."""
    settings = ws.settings
    if not moments:
        return []
    segments = load_transcript(ws.transcript_path) if ws.transcript_path.exists() else []
    jobs: list[tuple[Moment, Path]] = []
    for moment in moments:
        start = max(0.0, moment.start - settings.clip_padding)
        end = moment.end + settings.clip_padding
        if ws.vod.duration:
            end = min(end, ws.vod.duration)
        if end - start < settings.min_clip_length:
            end = start + settings.min_clip_length
        moment.start, moment.end = start, end
        jobs.append((moment, clip_output_path(ws.clips_dir, moment.index, moment.title, start)))

    def render(job: tuple[int, Moment, Path]) -> None:
        index, moment, path = job
        try:
            if ws.local_path is not None:
                produced = cut_clip_local(
                    ws.local_path,
                    path,
                    moment.start,
                    moment.end,
                    quality=settings.quality,
                    precise=settings.precise_cut,
                    overwrite=settings.force,
                )
                moment.clip_url = ""
            else:
                produced = cut_clip(
                    ws.vod.url,
                    path,
                    moment.start,
                    moment.end,
                    quality=settings.quality,
                    cookies_from_browser=settings.cookies_from_browser,
                    precise=settings.precise_cut,
                    overwrite=settings.force,
                )
                moment.clip_url = f"{ws.vod.url}?t={int(moment.start)}s"
            moment.clip_path = str(produced)
            if settings.burn_subs and segments:
                srt = write_clip_srt(
                    segments,
                    ws.clips_dir / f"{produced.stem}.srt",
                    moment.start,
                    moment.end,
                    use_translation=bool(settings.translate_to),
                )
                try:
                    produced = burn_subtitles(
                        produced, srt, ws.clips_dir / f"{produced.stem}_subs.mp4"
                    )
                    moment.clip_path = str(produced)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Клип {}: субтитры не вжглись ({})", produced.name, exc)
            if settings.vertical:
                try:
                    vertical = make_vertical(
                        produced, ws.clips_dir / f"{produced.stem}_vertical.mp4"
                    )
                    moment.clip_path = str(vertical)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Клип {}: вертикальная версия не сделана ({})", produced.name, exc)
            if settings.thumbnail:
                thumb = make_thumbnail(moment.clip_path or produced, ws.clips_dir / f"{path.stem}.jpg")
                moment.thumbnail_path = str(thumb)
        except Exception as exc:  # noqa: BLE001
            logger.error("Клип #{} не сделан: {}", index, exc)

    if len(jobs) > 20:
        logger.info("Клипов к нарезке: {} (примерно {} сек при 2 потоках)", len(jobs), len(jobs) * 20)
    enumerated = [(i, moment, path) for i, (moment, path) in enumerate(jobs, start=1)]
    with ThreadPoolExecutor(max_workers=max(1, int(settings.cut_workers))) as pool:
        list(pool.map(render, enumerated))
    logger.info("Нарезано клипов: {}", sum(1 for moment in moments if moment.clip_path))
    return list(moments)


def step_report(
    ws: Workspace,
    moments: Sequence[Moment],
    chat: Any = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Шаг 7: моментальный JSON + markdown + HTML-галерея."""
    summary_path = ws.root / "ai_summary.txt"
    ai_summary = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    chat_summary = None
    if chat is not None and hasattr(chat, "to_dict"):
        full = chat.to_dict()
        keep = {"total_messages", "unique_users", "duration", "emotes", "hype_words", "top_messages"}
        chat_summary = {key: value for key, value in full.items() if key in keep}
    return write_reports(
        ws.root,
        ws.vod,
        moments,
        ws.settings,
        chat_summary=chat_summary,
        ai_summary=ai_summary,
        extra=extra,
    )


def run_pipeline(
    target: str,
    settings: Settings,
    dump_chat: bool = False,
    clips: bool = True,
) -> dict[str, Any]:
    """Полный прогон: чат -> расшифровка -> детект -> клипы -> отчёт."""
    started = time.time()
    ws = prepare(target, settings)
    messages = step_chat(ws)
    if dump_chat:
        (ws.root / "chat.txt").write_text(
            "\n".join(
                f"[{fmt_ts(m.offset)}] {m.user}: {m.text}" for m in messages[:200000]
            ),
            encoding="utf-8",
        )
    step_transcribe(ws)
    moments, chat = step_detect(ws)
    if clips and moments:
        moments = step_clips(ws, moments)
    elapsed = time.time() - started
    extra = {
        "vod_id": ws.vod.id,
        "vod_url": ws.vod.url,
        "title": ws.vod.title,
        "duration": ws.vod.duration,
        "moments": len(moments),
        "chat_messages": len(messages),
        "elapsed_seconds": round(elapsed, 1),
        "settings": {
            "whisper_model": settings.whisper_model,
            "language": settings.language,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "quality": settings.quality,
            "max_clips": settings.max_clips,
            "clip_length": settings.clip_length,
            "limit_seconds": settings.limit_seconds,
        },
    }
    paths = step_report(ws, moments, chat, extra=extra)
    logger.info(
        "Готово за {}: {} моментов, клипы в {}",
        fmt_duration(elapsed),
        len(moments),
        ws.clips_dir,
    )
    return {
        "workspace": ws,
        "moments": moments,
        "paths": paths,
        "elapsed": elapsed,
        "chat": messages,
    }


def clip_timecodes(
    target: str,
    timecodes: Sequence[tuple[float, float, str]],
    settings: Settings,
) -> list[Moment]:
    """Ручная нарезка по заданным таймкодам (без ASR/детекта)."""
    ws = prepare(target, settings)
    moments: list[Moment] = []
    for index, (start, end, title) in enumerate(timecodes, start=1):
        moments.append(
            Moment(
                index=index,
                start=max(0.0, start - settings.clip_padding),
                end=end + settings.clip_padding,
                score=100.0,
                title=title or f"Клип {index}",
                category="manual",
                reason="Указано вручную в командной строке",
                source="manual",
            )
        )
    if ws.transcript_path.exists() or ws.wav_path.exists() or settings.force:
        # Расшифровка нужна, чтобы вжчь субтитры в ручные клипы (--burn-subs).
        step_transcribe(ws)
    moments = step_clips(ws, moments)
    step_report(ws, moments, None, extra={"vod_id": ws.vod.id, "manual": True})
    return moments


def load_moments(root: Path | str) -> list[Moment]:
    """Читает moments.json обратно в объекты Moment (для донарезки/пересборки отчёта)."""
    root = Path(root)
    raw = read_json(root / "moments.json", []) or []
    moments: list[Moment] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        moments.append(
            Moment(
                index=int(item.get("index") or index),
                start=float(item.get("start", 0.0)),
                end=float(item.get("end", 0.0)),
                score=float(item.get("score", 0.0)),
                title=str(item.get("title", "")),
                category=str(item.get("category", "other")),
                reason=str(item.get("reason", "")),
                source=str(item.get("source", "heuristic")),
                components=dict(item.get("components") or {}),
                chat_excerpt=list(item.get("chat_excerpt") or []),
                transcript=str(item.get("transcript", "")),
                clip_path=str(item.get("clip_path", "")),
                clip_url=str(item.get("clip_url", "")),
                thumbnail_path=str(item.get("thumbnail_path", "")),
            )
        )
    return moments


def rebuild_report(output_dir: Path | str, vod_id: str, settings: Settings) -> dict[str, str]:
    """Пересобирает отчёт из готовых moments.json (например, после ручной правки)."""
    root = Path(output_dir) / str(vod_id)
    meta = read_json(root / "metadata.json", {}) or {}
    vod = VodInfo(
        id=str(meta.get("id", vod_id)),
        title=meta.get("title", ""),
        duration=float(meta.get("duration", 0.0) or 0.0),
        channel=meta.get("channel", ""),
        channel_login=meta.get("channel_login", ""),
        game=meta.get("game", ""),
        created_at=meta.get("created_at", ""),
        url=meta.get("url", f"https://www.twitch.tv/videos/{vod_id}"),
        local=bool(meta.get("local", False)),
    )
    local_path = Path(vod.url) if vod.local and Path(vod.url).is_file() else None
    moments = load_moments(root)
    ws = Workspace(vod=vod, root=root, settings=settings, client=TwitchClient(delay=0.0), local_path=local_path)
    return step_report(ws, moments, None, extra={"vod_id": vod_id, "rebuilt": True})

