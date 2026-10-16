"""Общие утилиты: логирование, время, JSON/JSONL, запуск внешних процессов."""
from __future__ import annotations

import ctypes
import json
import os
import re
import subprocess
import sys
import unicodedata
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from loguru import logger

_LOGGING_READY = False


def setup_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    """Один раз настраивает loguru: цветной stderr + файл с UTF-8."""
    global _LOGGING_READY
    if _LOGGING_READY:
        return
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(str(log_file), level="DEBUG", encoding="utf-8", rotation="10 MB")
    _LOGGING_READY = True


def log() -> Any:
    """Доступ к логгеру без отдельного импорта loguru в модулях."""
    return logger


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def fmt_ts(seconds: float, with_ms: bool = False, force_hours: bool = False) -> str:
    """Секунды -> 'HH:MM:SS' (или 'MM:SS', если меньше часа).

    ``force_hours`` (мод toparachevt) печатает часы всегда: '0:04:19' вместо '04:19'.
    """
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    if hours or force_hours:
        text = f"{hours:d}:{minutes:02d}:{secs:02d}"
    else:
        text = f"{minutes:02d}:{secs:02d}"
    if with_ms:
        millis = int(round((seconds - int(seconds)) * 1000))
        text = f"{text}.{millis:03d}"
    return text


def fmt_duration(seconds: float) -> str:
    """Человекочитаемая длительность: 1 ч 23 мин 4 с."""
    seconds = int(max(0.0, seconds))
    if seconds < 60:
        return f"{seconds} с"
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    parts = []
    if hours:
        parts.append(f"{hours} ч")
    if minutes:
        parts.append(f"{minutes} мин")
    if secs and not hours:
        parts.append(f"{secs} с")
    return " ".join(parts)


def parse_ts(text: str) -> float:
    """'1:02:03' / '02:03' / '125' -> секунды."""
    text = str(text).strip()
    if not text:
        raise ValueError("пустой таймкод")
    parts = text.split(":")
    if len(parts) > 3:
        raise ValueError(f"некорректный таймкод: {text}")
    total = 0.0
    for part in parts:
        total = total * 60 + float(part)
    return total


def to_serializable(obj: Any) -> Any:
    """dataclass/np-объекты -> структуры, пригодные для json.dumps."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item") and callable(getattr(obj, "item")):  # numpy scalar
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            return str(obj)
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:  # noqa: BLE001
            return str(obj)
    return obj


def write_json(path: Path | str, data: Any, indent: int = 2) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(to_serializable(data), ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )
    return path


def read_json(path: Path | str, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        # utf-8-sig: терпим BOM, который добавляют Notepad/PowerShell 5.1
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return default


def write_jsonl(path: Path | str, rows: Iterable[Any]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(to_serializable(row), ensure_ascii=False) + "\n")
    return path


def append_jsonl(path: Path | str, rows: Iterable[Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(to_serializable(row), ensure_ascii=False) + "\n")


def iter_jsonl(path: Path | str) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def read_jsonl(path: Path | str) -> list[dict]:
    return list(iter_jsonl(path))


def chunked(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    size = max(1, int(size))
    for i in range(0, len(items), size):
        yield items[i : i + size]


def run_command(
    cmd: Sequence[str],
    cwd: Path | str | None = None,
    check: bool = True,
    quiet: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Запуск внешнего процесса с логом команды и хвостом вывода при ошибке."""
    logger.debug("run: {}", " ".join(str(c) for c in cmd))
    proc = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if proc.returncode != 0:
        if check:
            tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
            raise RuntimeError(
                f"Команда завершилась с кодом {proc.returncode}: {' '.join(str(c) for c in cmd)}\n{tail}"
            )
        logger.warning("команда вернула код {}: {}", proc.returncode, " ".join(str(c) for c in cmd))
    elif not quiet:
        logger.debug("ok: {}", " ".join(str(c) for c in cmd))
    return proc


def ffmpeg_exe() -> str:
    return "ffmpeg"


_DLL_DIR_HANDLES: list[Any] = []
_PRELOADED_DLLS: list[Any] = []

# Что CTranslate2 ищет сам; без предзагрузки её внутренний LoadLibrary не видит
# каталоги, добавленные через AddDllDirectory, и падает с «cublas64_12.dll is not found».
_CUDA_DLL_NAMES = ("cublas64_12.dll", "cublasLt64_12.dll")


def enable_cuda_dll_dirs(preload: bool = True) -> list[str]:
    """Регистрирует bin-каталоги pip-пакетов ``nvidia/*`` как источники CUDA-DLL (Windows).

    CTranslate2 (faster-whisper на GPU) грузит ``cublas64_12.dll``, но в системе часто
    стоит только CUDA 11 (например, ``cublas64_11.dll`` из torch). Ставим
    ``pip install nvidia-cublas-cu12`` — и DLL оттуда становится видна процессу.
    Вызывается один раз; возвращает список подключённых каталогов.
    """
    if os.name != "nt" or _DLL_DIR_HANDLES:
        return [str(getattr(handle, "path", "")) for handle in _DLL_DIR_HANDLES]
    import site

    roots: list[Path] = []
    for getter in (site.getsitepackages, site.getusersitepackages):
        try:
            found = getter()
        except Exception:  # noqa: BLE001
            continue
        roots.extend(Path(item) for item in (found if isinstance(found, (list, tuple)) else [found]))
    for root in dict.fromkeys(roots):
        nvidia_dir = root / "nvidia"
        if not nvidia_dir.is_dir():
            continue
        for bin_dir in sorted(nvidia_dir.glob("*/bin")):
            try:
                handle = os.add_dll_directory(str(bin_dir))
            except OSError:
                continue
            _DLL_DIR_HANDLES.append(handle)
            logger.debug("CUDA DLL: {}", bin_dir)
    if preload and not _PRELOADED_DLLS:
        for name in _CUDA_DLL_NAMES:
            try:
                _PRELOADED_DLLS.append(ctypes.WinDLL(name))
                logger.debug("CUDA DLL предзагружена: {}", name)
            except OSError:
                continue
    return [str(getattr(handle, "path", "")) for handle in _DLL_DIR_HANDLES]


def ffprobe_exe() -> str:
    return "ffprobe"


def ytdlp_cmd() -> list[str]:
    """yt-dlp запускаем как модуль текущего интерпретатора (CLI может быть не в PATH)."""
    return [sys.executable, "-m", "yt_dlp"]


_FILENAME_BAD = re.compile(r"[^\w\-. ]+", re.UNICODE)


def slugify(text: str, max_len: int = 60, fallback: str = "clip") -> str:
    """Безопасное имя файла из произвольного заголовка (кириллица сохраняется)."""
    text = unicodedata.normalize("NFKC", text or "").strip()
    text = _FILENAME_BAD.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip(" .-_")
    text = text.replace(" ", "_")
    if len(text) > max_len:
        text = text[:max_len].rstrip(" .-_")
    return text or fallback


def is_local_video(target: str) -> Path | None:
    """Если ``target`` — путь к существующему файлу на диске, вернуть его ``Path``.

    Позволяет вставлять локальное видео (``C:\\videos\\clip.mp4``, ``./stream.mkv``)
    вместо скачивания VOD по ссылке/ID Twitch. Кавычки, которыми Проводник Windows
    окружает скопированный путь, снимаются. ``None`` — это не локальный файл
    (значит, дальше разбираем как ID/ссылку/канал Twitch).
    """
    raw = (target or "").strip().strip('"').strip("'").strip()
    if not raw:
        return None
    try:
        candidate = Path(raw)
        if candidate.is_file():
            return candidate.resolve()
    except OSError:
        return None
    return None
