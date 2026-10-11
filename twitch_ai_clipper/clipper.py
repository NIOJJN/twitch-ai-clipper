"""Нарезка клипов: yt-dlp по секциям VOD + ffmpeg (вертикаль, субтитры, превью).

Скачивать весь VOD не нужно: yt-dlp умеет тянуть только нужный интервал
(``--download-sections``), поэтому клип на 40 секунд получается за секунды.
"""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from .utils import ensure_dir, fmt_ts, run_command, slugify, ytdlp_cmd

QUALITY_SELECTORS: dict[str, str] = {
    "best": "bv*+ba/b",
    "1080p60": "bv*[height<=1080][fps<=60]+ba/b[height<=1080]/b",
    "1080p": "bv*[height<=1080]+ba/b[height<=1080]/b",
    "720p60": "bv*[height<=720][fps<=60]+ba/b[height<=720]/b",
    "720p": "bv*[height<=720]+ba/b[height<=720]/b",
    "480p": "bv*[height<=480]+ba/b[height<=480]/b",
    "360p": "bv*[height<=360]+ba/b[height<=360]/b",
    "audio": "bestaudio/best",
}


def clip_output_path(out_dir: Path | str, index: int, title: str, start: float, ext: str = "mp4") -> Path:
    """Имя файла клипа: 0001_00-12-30_Название.mp4."""
    stamp = fmt_ts(start).replace(":", "-")
    return Path(out_dir) / f"{index:04d}_{stamp}_{slugify(title, max_len=48, fallback='moment')}.{ext}"


def cut_clip(
    vod_url: str,
    out_path: Path | str,
    start: float,
    end: float,
    quality: str = "1080p60",
    cookies_from_browser: str | None = None,
    precise: bool = False,
    overwrite: bool = False,
) -> Path:
    """Вырезает интервал [start, end] из VOD через yt-dlp (секции + ffmpeg)."""
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        logger.info("Клип уже есть: {}", out_path.name)
        return out_path
    ensure_dir(out_path.parent)
    selector = QUALITY_SELECTORS.get(quality, QUALITY_SELECTORS["best"])
    cmd = ytdlp_cmd() + [
        "--no-playlist",
        "--no-warnings",
        "--no-part",
        "-f", selector,
        "--download-sections", f"*{max(0.0, start):.2f}-{max(0.0, end):.2f}",
        "--merge-output-format", "mp4",
        "--remux-video", "mp4",
        "-o", str(out_path),
        "--newline",
    ]
    if precise:
        cmd += ["--force-keyframes-at-cuts"]
    if cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]
    cmd += [vod_url]
    logger.info(
        "Режем клип {} ({}) ...", out_path.name, f"{fmt_ts(start)}-{fmt_ts(end)}"
    )
    run_command(cmd, timeout=None)
    if not out_path.exists():
        candidates = sorted(out_path.parent.glob(out_path.stem + ".*"))
        if not candidates:
            raise RuntimeError(f"yt-dlp не создал клип {out_path}")
        out_path = candidates[0]
    logger.info("Клип готов: {} ({:.1f} МБ)", out_path.name, out_path.stat().st_size / 1_048_576)
    return out_path


def cut_clip_local(
    source: Path | str,
    out_path: Path | str,
    start: float,
    end: float,
    quality: str = "1080p60",
    precise: bool = False,
    overwrite: bool = False,
) -> Path:
    """Вырезает интервал [start, end] из локального видеофайла через ffmpeg (без сети)."""
    source = Path(source)
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        logger.info("Клип уже есть: {}", out_path.name)
        return out_path
    ensure_dir(out_path.parent)
    if not source.exists():
        raise RuntimeError(f"Локальный файл не найден: {source}")

    duration = max(0.1, end - start)
    # Для аудио дорожки перекодировать видео не нужно — копируем как есть.
    if quality == "audio":
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(0.0, start):.3f}", "-i", str(source),
            "-t", f"{duration:.3f}", "-vn",
            "-c:a", "libmp3lame", "-b:a", "192k", str(out_path),
        ]
    else:
        # --force-keyframes-at-cuts аналог для ffmpeg: точная перекодировка с ключевого
        # кадра вместо быстрого (но иногда неточного) копирования потока.
        codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-b:a", "160k"]
        if not precise:
            # Быстрый путь: ищем к ближайшему ключевому кадру и копируем потоки.
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{max(0.0, start):.3f}", "-i", str(source),
                "-t", f"{duration:.3f}",
                "-c", "copy", "-avoid_negative_ts", "make_zero", str(out_path),
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{max(0.0, start):.3f}", "-i", str(source),
                "-t", f"{duration:.3f}",
            ] + codec + [str(out_path)]
    logger.info(
        "Режем клип {} ({}) из локального файла ...", out_path.name, f"{fmt_ts(start)}-{fmt_ts(end)}"
    )
    run_command(cmd, timeout=None)
    if not out_path.exists():
        candidates = sorted(out_path.parent.glob(out_path.stem + ".*"))
        if not candidates:
            raise RuntimeError(f"ffmpeg не создал клип {out_path}")
        out_path = candidates[0]
    logger.info("Клип готов: {} ({:.1f} МБ)", out_path.name, out_path.stat().st_size / 1_048_576)
    return out_path


def make_vertical(source: Path | str, out_path: Path | str | None = None, overwrite: bool = False) -> Path:
    """9:16 для Shorts/TikTok: размытый фон + видео по центру."""
    source = Path(source)
    out_path = Path(out_path) if out_path else source.with_name(source.stem + "_vertical.mp4")
    if out_path.exists() and not overwrite:
        return out_path
    filter_complex = (
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,boxblur=24:2[bg];"
        "[0:v]scale=1080:-2[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]"
    )
    run_command(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "20", "-c:a", "aac", "-b:a", "160k", str(out_path),
        ]
    )
    return out_path


def burn_subtitles(source: Path | str, srt_path: Path | str, out_path: Path | str | None = None) -> Path:
    """Вжигает SRT в видео (запуск из папки клипа — так путь к файлу без проблем)."""
    source = Path(source).resolve()
    srt_path = Path(srt_path).resolve()
    out_path = Path(out_path) if out_path else source.with_name(source.stem + "_subs.mp4")
    style = (
        "FontName=Arial,FontSize=16,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H90000000,BorderStyle=1,Outline=2,Shadow=0,MarginV=36"
    )
    run_command(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", source.name,
            "-vf", f"subtitles={srt_path.name}:force_style='{style}'",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "copy", out_path.name,
        ],
        cwd=srt_path.parent,
    )
    return out_path


def make_thumbnail(source: Path | str, out_path: Path | str | None = None, at: float = 1.0) -> Path:
    """Превью клипа для отчёта."""
    source = Path(source)
    out_path = Path(out_path) if out_path else source.with_suffix(".jpg")
    run_command(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(0.0, at):.2f}", "-i", str(source), "-frames:v", "1",
            "-vf", "scale=640:-2", str(out_path),
        ],
        check=False,
    )
    return out_path
