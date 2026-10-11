"""CLI-интерфейс: run / chat / transcribe / detect / clip / report / list / serve."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from loguru import logger

from . import __version__
from .config import Settings, load_settings
from .detect import Moment
from .modes import MODES, apply_mode, get_mode
from .pipeline import (
    clip_timecodes,
    load_moments,
    prepare,
    rebuild_report,
    run_pipeline,
    step_clips,
    step_detect,
    step_report,
    step_transcribe,
)
from .twitch_api import TwitchClient
from .utils import fmt_duration, fmt_ts, parse_ts, setup_logging

QUALITY_CHOICES = ["best", "1080p60", "1080p", "720p60", "720p", "480p", "360p", "audio"]

COMMON_DEFAULTS: dict[str, Any] = {
    "whisper_model": None,
    "language": None,
    "llm_provider": None,
    "llm_model": None,
    "quality": None,
    "output_dir": None,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="twitch-clip",
        description=(
            "AI-нарезка стримов Twitch: скачивает VOD и чат, расшифровывает речь "
            "(faster-whisper), находит хайлайты (чат + звук + LLM) и режет клипы."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python -m twitch_ai_clipper run twitch.tv/tarik\n"
            "  python -m twitch_ai_clipper run https://www.twitch.tv/videos/123456789 "
            "--whisper-model large-v3 --language ru --max-clips 10\n"
            "  python -m twitch_ai_clipper run twitch.tv/канал --mode toparachevt\n"
            "  python -m twitch_ai_clipper chat 123456789\n"
            "  python -m twitch_ai_clipper detect 123456789 --llm ollama --llm-model qwen2.5:7b\n"
            "  python -m twitch_ai_clipper clip 123456789 --timecodes 0:10-0:45=Начало\n"
            "  python -m twitch_ai_clipper serve --port 8765\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"twitch-ai-clipper {__version__}")
    parser.add_argument("--config", help="путь к config.toml (по умолчанию ./config.toml)")
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог")
    sub = parser.add_subparsers(dest="command", required=True)
    return sub, parser


def add_common(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("target", help="ID VOD, ссылка на VOD, имя канала ИЛИ путь к локальному видеофайлу (например C:\\videos\\stream.mp4)")
    sub.add_argument("-o", "--output-dir", help="папка результатов (по умолчанию output/)")
    sub.add_argument("--whisper-model", help="модель Whisper (tiny/base/small/medium/large-v3) или путь к локальной CT2-модели")
    sub.add_argument("--asr-backend", choices=["auto", "faster-whisper", "openai-whisper"], help="движок распознавания речи")
    sub.add_argument("--strict-asr", dest="asr_optional", action="store_false", help="падать с ошибкой, если модель не загрузилась")
    sub.add_argument("--models-dir", help="куда складывать/откуда брать веса моделей (по умолчанию models/)")
    sub.add_argument("--hf-endpoint", help="зеркало HuggingFace, например https://hf-mirror.com")
    sub.add_argument("--device", help="auto|cuda|cpu")
    sub.add_argument("--compute-type", help="auto|float16|int8_float16|int8|float32")
    sub.add_argument("--language", help="язык стрима (ru/en/...); по умолчанию ru, auto — определять самому")
    sub.add_argument("--translate-to", help="перевести расшифровку через LLM на этот язык")
    sub.add_argument("--llm", dest="llm_provider", help="auto|none|ollama|openai")
    sub.add_argument("--llm-model", help="имя модели (например qwen2.5:7b-instruct)")
    sub.add_argument("--llm-base-url", help="URL API (Ollama или OpenAI-совместимый)")
    sub.add_argument("--llm-api-key", help="ключ API (или переменная OPENAI_API_KEY)")
    sub.add_argument("--quality", choices=QUALITY_CHOICES, help="качество клипов")
    sub.add_argument("--cookies-from-browser", help="взять cookies из браузера (chrome/firefox/edge)")
    sub.add_argument("--limit-seconds", type=float, help="анализировать только первые N секунд (тесты)")
    sub.add_argument("--force", action="store_true", default=None, help="игнорировать кеш и перегенерировать всё")


def add_mode(sub: argparse.ArgumentParser) -> None:
    """Флаг --mode: оформление таймкодов и раскладка детекта."""
    modes_help = "; ".join(f"{m.name} — {m.description}" for m in MODES.values())
    sub.add_argument(
        "--mode",
        metavar="MODE",
        help=f"{modes_help}. Алиас: топарашевт",
    )



PRESETS: dict[str, dict[str, Any]] = {
    # Прореженный «журнал»: 15 сильнейших моментов.
    "sparse": {
        "max_clips": 15, "min_moments": 0, "min_gap": 45.0, "top_candidates": 40,
        "bucket": 5.0, "min_clip_length": 12.0,
    },
    # Много моментов: ~1 клип в минуту стрима.
    "dense": {
        "max_clips": 150, "min_moments": 100, "min_gap": 12.0, "top_candidates": 400,
        "bucket": 5.0, "min_clip_length": 10.0, "max_clip_length": 90.0,
    },
    # Подробный разбор: всё, что вообще нашлось, короткими кусками.
    "full": {
        "max_clips": 600, "min_moments": 300, "min_gap": 5.0, "top_candidates": 1500,
        "bucket": 3.0, "min_clip_length": 8.0, "max_clip_length": 60.0,
    },
}


def apply_preset(name: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """Раскладывает пресет в overrides (явные флаги CLI имеют приоритет)."""
    preset = PRESETS.get((name or "").lower())
    if not preset:
        return overrides
    for key, value in preset.items():
        if key not in overrides:
            overrides[key] = value
    return overrides


def build_subcommands(sub: Any) -> None:
    run = sub.add_parser("run", help="полный прогон: чат + расшифровка + детект + клипы + отчёт")
    add_common(run)
    run.add_argument("--bucket", type=float, help="окно агрегации сигналов, сек (по умолчанию 5)")
    run.add_argument("--clip-length", type=float, help="целевая длина клипа, сек")
    run.add_argument("--max-clips", type=int, help="максимум клипов")
    run.add_argument(
        "--min-moments",
        type=int,
        help="гарантированно выдать не меньше N моментов (добирает, ослабляя отбор)",
    )
    run.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        help="sparse=15 сильных | dense=~1 клип/мин (min 100) | full=максимально подробно",
    )
    run.add_argument("--min-clip-length", type=float, help="минимальная длина клипа, сек")
    run.add_argument("--max-clip-length", type=float, help="максимальная длина клипа, сек")
    run.add_argument("--min-gap", type=float, help="мин. расстояние между клипами, сек")
    run.add_argument("--top-candidates", type=int, help="сколько кандидатов брать")
    run.add_argument("--clip-padding", type=float, help="запас до/после момента, сек")
    run.add_argument("--vertical", action="store_true", default=None, help="сделать вертикальные версии (9:16)")
    run.add_argument("--burn-subs", action="store_true", default=None, help="вжечь субтитры в клипы")
    run.add_argument("--precise-cut", action="store_true", default=None, help="точная резка по кадрам (медленнее)")
    run.add_argument("--no-clips", action="store_true", default=None, help="только анализ, без нарезки")
    run.add_argument("--dump-chat", action="store_true", default=None, help="дополнительно сохранить chat.txt")
    add_mode(run)
    run.add_argument("--chat-stride", type=float, help="шаг сканирования чата в тихих местах, сек")

    chat = sub.add_parser("chat", help="только скачать чат VOD в JSONL")
    add_common(chat)
    chat.add_argument("--chat-stride", type=float, help="шаг сканирования в тихих местах, сек")
    chat.add_argument("--chat-delay", type=float, help="пауза между запросами к Twitch, сек")
    chat.add_argument("--dump-chat", action="store_true", default=None, help="сохранить читаемый chat.txt")

    asr = sub.add_parser("transcribe", help="скачать аудио и расшифровать речь")
    add_common(asr)

    detect = sub.add_parser("detect", help="найти моменты (нужны chat.jsonl и transcript.jsonl)")
    add_common(detect)
    detect.add_argument("--bucket", type=float, help="окно агрегации сигналов, сек")
    detect.add_argument("--max-clips", type=int, help="максимум клипов")
    detect.add_argument("--min-moments", type=int, help="гарантированно выдать не меньше N моментов")
    detect.add_argument("--preset", choices=sorted(PRESETS), help="sparse | dense | full")
    detect.add_argument("--min-clip-length", type=float, help="минимальная длина клипа, сек")
    detect.add_argument("--max-clip-length", type=float, help="максимальная длина клипа, сек")
    detect.add_argument("--top-candidates", type=int, help="сколько кандидатов брать до отбора")
    detect.add_argument("--min-gap", type=float, help="мин. расстояние между клипами, сек")
    detect.add_argument("--clip-length", type=float, help="целевая длина клипа, сек")
    detect.add_argument("--no-clips", action="store_true", default=None, help="только моменты, без нарезки")
    add_mode(detect)

    clip = sub.add_parser("clip", help="нарезать клипы: по moments.json или по своим таймкодам")
    add_common(clip)
    clip.add_argument(
        "--timecodes",
        nargs="*",
        help="ручные таймкоды вида 0:10-0:45=Заголовок (можно несколько)",
    )
    clip.add_argument(
        "--timecodes-file",
        help="файл со списком глав, по строке: '2:04:19 - Название' (старт = начало главы)",
    )
    clip.add_argument("--vertical", action="store_true", default=None, help="вертикальные версии (9:16)")
    clip.add_argument("--burn-subs", action="store_true", default=None, help="вжечь субтитры из транскрипта")
    clip.add_argument("--precise-cut", action="store_true", default=None, help="точная резка по кадрам")
    clip.add_argument("--only", nargs="*", type=int, help="номера моментов из moments.json")
    add_mode(clip)

    report = sub.add_parser("report", help="пересобрать report.md/index.html из moments.json")
    report.add_argument("vod_id", help="ID VOD (папка в --output-dir)")
    report.add_argument("-o", "--output-dir", help="папка результатов")
    add_mode(report)

    listing = sub.add_parser("list", help="показать последние записи канала")
    listing.add_argument("channel", help="имя канала")
    listing.add_argument("-n", "--limit", type=int, default=10, help="сколько записей показать")

    serve = sub.add_parser("serve", help="запустить локальную галерею результатов")
    serve.add_argument("-o", "--output-dir", help="папка результатов")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)


# '2:04:19 - Название' | '2:04:19-2:06:03=Название' | '125=Название'
_TIMECODE_RE = re.compile(
    r"^\s*(?P<start>\d{1,2}:\d{2}(?::\d{2})?|\d+(?:\.\d+)?)\s*(?:(?P<dash>-)\s*(?P<rest>.*?))?\s*$"
)
_STAMP_RE = re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$|^\d+(?:\.\d+)?$")


def parse_timecode_line(
    line: str, default_length: float = 40.0
) -> tuple[float, float, str] | None:
    """Одна строка таймкода -> (start, end, title). ``None`` — пусто/комментарий.

    Понимает форматы модов вроде toparachevt::

        2:04:19 - Любимый цвет        # только начало, конец = start + default_length
        2:18:21- Мазелов              # без пробела тоже
        2:04:19-2:06:03=Любимый цвет  # явный интервал
    """
    text = (line or "").strip()
    if not text or text.startswith("#"):
        return None
    if "=" in text:  # заголовок после '=' (тогда дефис в заголовке не мешает)
        body, title = text.split("=", 1)
    else:
        body, title = text, ""
    match = _TIMECODE_RE.match(body)
    if not match:
        raise ValueError(
            f"Некорректный таймкод: {line!r} "
            "(ожидается '2:04:19 - Название' или '2:04:19-2:06:03=Название')"
        )
    start = parse_ts(match.group("start"))
    rest = (match.group("rest") or "").strip()
    if match.group("dash") and rest and _STAMP_RE.match(rest):
        end = parse_ts(rest)
        if end <= start:
            raise ValueError(f"Конец раньше начала: {line!r}")
    else:
        # Только начало: дефис отделён пробелом или его нет вовсе.
        end = start + max(1.0, float(default_length))
        if match.group("dash") and rest and not title:
            title = rest
    return start, end, (title or rest or "").strip()


def parse_timecodes(
    raw: list[str] | None, default_length: float = 40.0
) -> list[tuple[float, float, str]]:
    """'0:10-0:45=Заголовок' -> (start, end, title)."""
    result: list[tuple[float, float, str]] = []
    for item in raw or []:
        parsed = parse_timecode_line(item, default_length)
        if parsed is not None:
            result.append(parsed)
    return result


def parse_timecodes_file(
    path: Path | str, default_length: float = 40.0
) -> list[tuple[float, float, str]]:
    """Список глав из файла (по строке на главу, как timecodes.txt мода toparachevt)."""
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError(f"Файл с таймкодами не найден: {file_path}")
    result: list[tuple[float, float, str]] = []
    for line in file_path.read_text(encoding="utf-8-sig").splitlines():
        parsed = parse_timecode_line(line, default_length)
        if parsed is not None:
            result.append(parsed)
    if not result:
        raise ValueError(f"В файле {file_path} не нашлось ни одного таймкода")
    return result



def settings_from_args(args: argparse.Namespace) -> Settings:
    """CLI-аргументы -> Settings (заданные значения перекрывают config.toml)."""
    from dataclasses import fields

    valid = {field.name for field in fields(Settings)}
    overrides: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key not in valid or value is None:
            continue
        overrides[key] = value
    if getattr(args, "device", None):
        overrides["whisper_device"] = args.device
    if getattr(args, "compute_type", None):
        overrides["whisper_compute_type"] = args.compute_type
    apply_preset(getattr(args, "preset", ""), overrides)
    apply_mode(getattr(args, "mode", ""), overrides)
    return load_settings(getattr(args, "config", None), overrides)


def _print_moments(moments: list[Moment], mode: str = "default") -> None:
    if not moments:
        print("Моменты не найдены.")
        return
    resolved = get_mode(mode)
    if resolved.chapter_list:
        # Мод toparachevt: список глав в его собственном оформлении.
        print(f"\n{resolved.title} — оглавление ({len(moments)} глав):\n")
        for moment in moments:
            print(moment.chapter(resolved.name))
        print()
        return
    print(f"\n{'#':>3}  {'таймкод':>9}  {'score':>5}  {'тип':<10}  название")
    print("-" * 90)
    for moment in moments:
        print(
            f"{moment.index:>3}  {fmt_ts(moment.start):>9}  {moment.score:>5.0f}  "
            f"{moment.category:<10}  {moment.title[:58]}"
        )
    print()


def command_chat(args: argparse.Namespace, settings: Settings) -> int:
    from .pipeline import step_chat

    ws = prepare(args.target, settings)
    messages = step_chat(ws)
    print(f"Чат сохранён: {ws.chat_path} ({len(messages)} сообщений)")
    if getattr(args, "dump_chat", False):
        text_path = ws.root / "chat.txt"
        text_path.write_text(
            "\n".join(f"[{fmt_ts(m.offset)}] {m.user}: {m.text}" for m in messages),
            encoding="utf-8",
        )
        print(f"Читаемый лог: {text_path}")
    return 0


def command_transcribe(args: argparse.Namespace, settings: Settings) -> int:
    from .pipeline import step_audio

    ws = prepare(args.target, settings)
    step_audio(ws)
    result = step_transcribe(ws)
    print(
        f"Расшифровка: {len(result.segments)} реплик, язык {result.language or '?'}, "
        f"устройство {result.device}/{result.compute_type}, "
        f"{fmt_duration(result.elapsed)} на {fmt_ts(result.duration)} аудио"
    )
    print(f"Файлы: {ws.transcript_path}, transcript.srt, transcript.txt")
    return 0


def command_detect(args: argparse.Namespace, settings: Settings) -> int:
    ws = prepare(args.target, settings)
    moments, chat = step_detect(ws)
    if moments:
        # Пересобираем отчёт и галерею сразу — иначе они останутся от прошлого прогона.
        step_report(
            ws,
            moments,
            chat,
            extra={
                "vod_id": ws.vod.id,
                "moments": len(moments),
                "clips_cut": False,
                "settings": {
                    "whisper_model": settings.whisper_model,
                    "llm_provider": settings.llm_provider,
                    "min_moments": settings.min_moments,
                    "max_clips": settings.max_clips,
                    "min_gap": settings.min_gap,
                    "min_clip_length": settings.min_clip_length,
                },
                "mode": settings.mode,
            },
        )
    _print_moments(moments, settings.mode)
    print(f"moments.json: {ws.moments_path}")
    print(f"Отчёт:        {ws.root / 'report.md'}")
    print(f"Галерея:      {ws.root / 'index.html'}")
    if get_mode(settings.mode).chapter_list:
        print(f"Таймкоды:     {ws.root / 'timecodes.txt'}")
    pending = sum(1 for moment in moments if not moment.clip_path)
    if pending:
        print(f"\nКлипов ещё не нарезано: {pending}. Создать их: run.bat clip {ws.vod.id}")
    return 0


def command_clip(args: argparse.Namespace, settings: Settings) -> int:
    default_length = max(5.0, float(settings.clip_length))
    manual = parse_timecodes(getattr(args, "timecodes", None), default_length)
    timecodes_file = getattr(args, "timecodes_file", None)
    if timecodes_file:
        from_file = parse_timecodes_file(timecodes_file, default_length)
        print(f"Из файла {timecodes_file} взято глав: {len(from_file)}")
        manual += from_file
    if manual:
        moments = clip_timecodes(args.target, manual, settings)
        _print_moments(moments, settings.mode)
        return 0
    ws = prepare(args.target, settings)
    all_moments = load_moments(ws.root)
    only = getattr(args, "only", None)
    # --only режет клипы лишь часть глав, но moments.json/timecodes.txt/report
    # должны остаться полными, иначе оглавление теряет невыбранные главы.
    targets = all_moments
    if only:
        wanted = set(only)
        targets = [moment for moment in all_moments if moment.index in wanted]
    if not targets:
        print("Нет моментов для нарезки: сначала выполните 'detect' или передайте --timecodes")
        return 1
    step_clips(ws, targets)
    # Клипы нарезаны только у targets — сшиваем обратно полный список,
    # чтобы в отчёте и moments.json остались все главы (у остальных без clip_path).
    by_index = {moment.index: moment for moment in targets}
    for position, moment in enumerate(all_moments, start=1):
        updated = by_index.get(moment.index)
        if updated is not None:
            moment.clip_path = updated.clip_path
            moment.clip_url = updated.clip_url
            moment.thumbnail_path = updated.thumbnail_path
            moment.start, moment.end = updated.start, updated.end
    step_report(ws, all_moments, None, extra={"vod_id": ws.vod.id, "recut": True, "only": only})
    _print_moments(targets, settings.mode)
    print(f"Клипы: {ws.clips_dir}")
    if len(targets) < len(all_moments):
        print(f"Нарезано {len(targets)} из {len(all_moments)} глав; остальные — в {ws.moments_path}")
    return 0



def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI."""
    sub, parser = build_parser()
    build_subcommands(sub)
    args = parser.parse_args(argv)
    setup_logging(verbose=bool(getattr(args, "verbose", False)))
    try:
        settings = settings_from_args(args)
    except ValueError as exc:  # например, неизвестный --mode
        print(f"Ошибка: {exc}")
        return 2
    if settings.config_file:
        logger.debug("Используется конфиг: {}", settings.config_file)
    try:
        if args.command == "list":
            return command_list(args)
        if args.command == "serve":
            from .webui import serve

            serve(settings.output_dir, host=args.host, port=args.port)
            return 0
        if args.command == "chat":
            return command_chat(args, settings)
        if args.command == "transcribe":
            return command_transcribe(args, settings)
        if args.command == "detect":
            return command_detect(args, settings)
        if args.command == "clip":
            return command_clip(args, settings)
        if args.command == "report":
            paths = rebuild_report(Path(settings.output_dir), args.vod_id, settings)
            print("Отчёт пересобран:", paths["report"])
            print("Галерея:", paths["html"])
            if paths.get("timecodes"):
                print("Таймкоды:", paths["timecodes"])
            return 0
        if args.command == "run":
            result = run_pipeline(
                args.target,
                settings,
                dump_chat=bool(getattr(args, "dump_chat", False)),
                clips=not bool(getattr(args, "no_clips", False)),
            )
            moments: list[Moment] = result["moments"]
            ws = result["workspace"]
            _print_moments(moments, settings.mode)
            print(f"Папка VOD: {ws.root}")
            print(f"Отчёт:     {result['paths']['report']}")
            print(f"Галерея:   {result['paths']['html']}")
            if result["paths"].get("timecodes"):
                print(f"Таймкоды:  {result['paths']['timecodes']}")
            print(f"Клипы:     {ws.clips_dir} ({sum(1 for m in moments if m.clip_path)} шт.)")
            print(f"Время:     {fmt_duration(result['elapsed'])}")
            if result.get("paths", {}).get("run_summary"):
                logger.debug("Сводка: {}", result["paths"]["run_summary"])
            return 0
    except KeyboardInterrupt:
        logger.warning("Остановлено пользователем")
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.error("Ошибка: {}", exc)
        if getattr(args, "verbose", False):
            raise
        return 1
    parser.print_help()
    return 1


def command_list(args: argparse.Namespace) -> int:
    client = TwitchClient()
    for video in client.list_videos(args.channel, limit=args.limit):
        length = float(video.get("lengthSeconds") or 0)
        print(
            f"{video.get('id')}  {fmt_ts(length):>9}  {video.get('publishedAt', '')[:10]}  "
            f"{(video.get('title') or '')[:70]}"
        )
    return 0
