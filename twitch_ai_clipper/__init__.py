"""twitch_ai_clipper -- AI-инструмент для поиска хайлайтов и нарезки стримов Twitch.

Пайплайн:
    metadata -> chat download -> audio -> ASR (faster-whisper) -> signals
             -> moment detection (эвристика + LLM) -> clips (ffmpeg/yt-dlp) -> report
"""
from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
