"""Быстрая проверка Twitch-слоя на живом VOD (метаданные + чат).

Запуск: python -u scripts/smoke_api.py [video_id] [window_seconds]
Вывод пишется в probe_out/smoke.log (в нём же лог клиента).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from twitch_ai_clipper.twitch_api import TwitchClient  # noqa: E402
from twitch_ai_clipper.utils import setup_logging  # noqa: E402


def main() -> None:
    video_id = sys.argv[1] if len(sys.argv) > 1 else "2840912067"
    window = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    out_dir = ROOT / "probe_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(verbose=False, log_file=out_dir / "smoke.log")
    client = TwitchClient(delay=0.15)

    started = time.time()
    vod = client.get_vod(video_id)
    print(f"VOD: {vod.id} | {vod.title[:70]} | {vod.duration:.0f}s | {vod.channel} | {vod.game}")
    print(f"meta latency: {time.time() - started:.2f}s")

    started = time.time()
    page = client.comments_page(vod.id, 0)
    print(f"page[0]: {len(page)} сообщений за {time.time() - started:.2f}s")
    if page:
        print("sample:", json.dumps(page[0].to_dict(), ensure_ascii=False)[:400])
        print("offsets:", [m.offset for m in page[:4]], "...", [m.offset for m in page[-3:]])
        with_emotes = [m for m in page if m.emotes]
        print(f"с эмоутами: {len(with_emotes)} шт, примеры:", [m.text for m in with_emotes[:6]])

    started = time.time()
    stats = client.download_chat(
        vod.id,
        out_dir / "chat_test.jsonl",
        duration=window,
        stride=45.0,
        resume=False,
        gap_audit=True,
    )
    print("download:", json.dumps(stats, ensure_ascii=False))
    print(f"chat latency: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
