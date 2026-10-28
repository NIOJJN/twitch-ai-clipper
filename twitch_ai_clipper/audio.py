"""Аудио: скачивание дорожки через yt-dlp, конвертация в WAV 16 kHz mono и расчёт энергии.

Энергия/динамика звука — второй по важности сигнал после чата: крики, смех,
взрывы в игре и громкая музыка прекрасно ловятся простым RMS.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from loguru import logger

from .utils import ensure_dir, run_command, ytdlp_cmd

AUDIO_SELECTOR = "bestaudio/best"
PARTIAL_SUFFIXES = (".temp", ".part", ".ytdl")


def probe_duration(path: Path | str) -> float:
    """Длительность медиафайла через ffprobe."""
    proc = run_command(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        check=False,
        quiet=True,
    )
    try:
        return float(json.loads(proc.stdout)["format"]["duration"])
    except Exception:  # noqa: BLE001
        return 0.0


def is_partial(path: Path) -> bool:
    """Недоскачанный кусок: audio.temp.mp4, audio.mp4.part и т.п."""
    name = path.name.lower()
    return any(token in name for token in PARTIAL_SUFFIXES)


def pick_audio_file(work_dir: Path | str) -> Path | None:
    """Готовый (не частичный) аудиофайл с наибольшей длительностью."""
    work_dir = Path(work_dir)
    candidates = [
        path
        for path in sorted(work_dir.glob("audio*"))
        if path.is_file() and path.suffix.lower() not in {".wav", ".srt", ".json", ".txt"} and not is_partial(path)
    ]
    if not candidates:
        return None
    best, best_duration = None, -1.0
    for path in candidates:
        duration = probe_duration(path)
        if duration > best_duration:
            best, best_duration = path, duration
    return best


def clear_partials(work_dir: Path | str, attempts: int = 3) -> None:
    """Удаляет недоскачанные остатки (Windows может держать файл — пробуем несколько раз)."""
    work_dir = Path(work_dir)
    for path in sorted(work_dir.glob("audio*")):
        if not path.is_file():
            continue
        if not (is_partial(path) or path.suffix.lower() in {".f137", ".f140"}):
            continue
        for attempt in range(attempts):
            try:
                path.unlink()
                logger.info("Удалён мусор от прошлой загрузки: {}", path.name)
                break
            except PermissionError:
                time.sleep(1.0 + attempt)
            except OSError as exc:  # noqa: PERF203
                logger.debug("Не удалось удалить {}: {}", path.name, exc)
                break


def _move_with_retry(source: Path, target: Path, attempts: int = 5) -> bool:
    """Переименование/копирование с ретраями: Windows (антивирус, Проводник) держит файлы."""
    if source == target:
        return True
    for attempt in range(attempts):
        try:
            if target.exists():
                try:
                    target.unlink()
                except OSError:
                    shutil.copy2(source, target)
                    return True
            os.replace(source, target)
            return True
        except PermissionError:
            logger.debug("Файл занят ({}), пробуем ещё раз: {}", attempt + 1, target.name)
            time.sleep(1.0 + attempt * 1.5)
        except OSError as exc:
            logger.debug("Не удалось переместить {}: {}", source.name, exc)
            time.sleep(1.0)
    return False


def download_audio(
    vod_url: str,
    work_dir: Path | str,
    limit_seconds: float = 0.0,
    cookies_from_browser: str | None = None,
    retries: int = 3,
    expected_duration: float = 0.0,
) -> Path:
    """Скачивает аудиодорожку VOD в ``work_dir``; возвращает путь к готовому файлу.

    Устойчив к типичной проблеме Windows: yt-dlp скачивает в ``audio.temp.mp4``,
    но не может переименовать его в ``audio.mp4`` (``WinError 5`` — файл держит
    антивирус/Проводник). В таком случае докачанный временный файл всё равно
    полноценен, поэтому мы аккуратно переносим его на место.
    """
    work_dir = ensure_dir(work_dir)

    existing = pick_audio_file(work_dir)
    if existing is not None and _is_complete(existing, expected_duration):
        logger.info("Аудио уже скачано: {} ({:.1f} МБ)", existing.name, existing.stat().st_size / 1_048_576)
        return existing

    clear_partials(work_dir)
    # Старый файл может быть «в работе» другой программой — уводим его в сторону.
    if existing is not None:
        if not _move_with_retry(existing, existing.with_suffix(f".old{existing.suffix}"), attempts=2):
            logger.warning("Не могу освободить {} — загружу под новым именем", existing.name)
        clear_partials(work_dir)

    template = str(work_dir / "audio.%(ext)s")
    cmd = ytdlp_cmd() + [
        "--no-playlist",
        "--no-warnings",
        "--no-part",
        "--retries", str(retries),
        "--fragment-retries", str(retries),
        "-f", AUDIO_SELECTOR,
        "-o", template,
    ]
    if limit_seconds and limit_seconds > 0:
        cmd += ["--download-sections", f"*0-{int(limit_seconds)}", "--force-keyframes-at-cuts"]
    if cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]
    cmd += [vod_url]

    logger.info("Скачиваем аудио: {} ...", vod_url)
    proc = run_command(cmd, timeout=None, check=False)
    if proc.returncode != 0:
        logger.warning("yt-dlp завершился с кодом {} — пробуем спасти докачанный файл", proc.returncode)
        recovered = _recover_partial(work_dir)
        if recovered is not None and _is_complete(recovered, expected_duration):
            logger.info("Восстановлен скачанный аудиофайл: {}", recovered.name)
            return recovered
        detail = (proc.stderr or proc.stdout or "").strip()[-400:]
        raise RuntimeError(
            "Не удалось скачать аудио VOD.\n"
            f"{detail}\n\n"
            "Чаще всего это блокировка файла на Windows (антивирус/Проводник/плеер). Что делать:\n"
            "  • закройте Проводник с папкой output, видеоплееры и проигрыватели;\n"
            "  • удалите папку output\\<vod_id>\\work и запустите команду снова;\n"
            "  • при желании добавьте папку проекта в исключения Защитника Windows.\n"
            "  • если файл уже скачан (в work\\ лежит audio.mp4), просто запустите команду повторно — "
            "он будет использован без повторной загрузки."
        )

    produced = pick_audio_file(work_dir)
    if produced is None:
        produced = _recover_partial(work_dir)
    if produced is None:
        raise RuntimeError("yt-dlp отработал без ошибки, но аудиофайл не найден в " + str(work_dir))
    if not _is_complete(produced, expected_duration):
        logger.warning(
            "Аудио короче ожидаемого: {:.0f} с из {:.0f} с. Повторите запуск — докачается.",
            probe_duration(produced), expected_duration,
        )
    logger.info("Аудио готово: {} ({:.1f} МБ)", produced.name, produced.stat().st_size / 1_048_576)
    return produced


def _is_complete(path: Path, expected_duration: float) -> bool:
    """Файл покрывает ожидаемую длительность (с допуском 3 %)."""
    if not expected_duration or expected_duration <= 0:
        return path.stat().st_size > 0
    actual = probe_duration(path)
    return actual > 0 and actual >= expected_duration * 0.97


def _recover_partial(work_dir: Path) -> Path | None:
    """Из audio.temp.mp4 / audio.mp4.part делает полноценный audio.mp4."""
    work_dir = Path(work_dir)
    partials = [p for p in sorted(work_dir.glob("audio*")) if p.is_file() and is_partial(p)]
    for partial in partials:
        suffix = partial.name.split(".", 1)[1] if "." in partial.name else "mp4"
        # audio.temp.mp4 -> "temp.mp4"; audio.mp4.part -> "mp4.part".
        # Снимаем маркеры недокачки: ведущий "temp." и хвостовой ".part"/".ytdl".
        # Иначе target == partial и _move_with_retry удалит файл вместо спасения.
        suffix = suffix.replace("temp.", "")
        for marker in (".part", ".ytdl"):
            if suffix.endswith(marker):
                suffix = suffix[: -len(marker)]
        target = work_dir / f"audio.{suffix or 'mp4'}"
        if target == partial:
            continue
        if _move_with_retry(partial, target):
            logger.info("Восстановлено: {} -> {}", partial.name, target.name)
            return target
        # Последний шанс: скопировать вместо переименования.
        try:
            shutil.copy2(partial, target)
            partial.unlink(missing_ok=True)
            logger.info("Скопировано: {} -> {}", partial.name, target.name)
            return target
        except OSError as exc:  # noqa: PERF203
            logger.warning("Восстановить {} не вышло: {}", partial.name, exc)
    return None


def to_wav(
    source: Path | str,
    out_wav: Path | str,
    sample_rate: int = 16000,
    start: float | None = None,
    duration: float | None = None,
    force: bool = False,
) -> Path:
    """Конвертирует произвольный медиафайл в WAV mono ``sample_rate`` (для Whisper)."""
    source = Path(source)
    out_wav = Path(out_wav)
    if out_wav.exists() and not force:
        return out_wav
    ensure_dir(out_wav.parent)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if start:
        cmd += ["-ss", f"{float(start):.3f}"]
    cmd += ["-i", str(source)]
    if duration:
        cmd += ["-t", f"{float(duration):.3f}"]
    cmd += [
        "-vn", "-ac", "1", "-ar", str(int(sample_rate)),
        "-c:a", "pcm_s16le", "-f", "wav", str(out_wav),
    ]
    run_command(cmd)
    return out_wav


def audio_rms(wav_path: Path | str, hop: float = 1.0) -> np.ndarray:
    """RMS-громкость по окнам ``hop`` секунд в дБFS (потоковое чтение, экономит RAM)."""
    import soundfile as sf

    path = str(wav_path)
    info = sf.info(path)
    block = max(1, int(info.samplerate * hop))
    values: list[float] = []
    with sf.SoundFile(path) as handle:
        while True:
            data = handle.read(block, dtype="float32", always_2d=False)
            if data is None or len(data) == 0:
                break
            if getattr(data, "ndim", 1) > 1:
                data = data.mean(axis=1)
            values.append(float(np.sqrt(np.mean(np.square(data)) + 1e-12)))
    rms = np.asarray(values, dtype=np.float64)
    if rms.size == 0:
        return rms
    return 20.0 * np.log10(rms + 1e-9)


def audio_dynamics(db: np.ndarray, hop: float = 1.0) -> np.ndarray:
    """Модуль производной громкости — «внезапный шум» (крик, смех, взрыв)."""
    if db.size == 0:
        return db
    diff = np.abs(np.diff(db, prepend=db[0]))
    window = max(1, int(round(1.0 / hop)))
    if window > 1 and diff.size >= window:
        kernel = np.ones(window) / window
        diff = np.convolve(diff, kernel, mode="same")
    return diff
