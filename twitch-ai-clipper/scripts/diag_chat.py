"""Диагностика доступности чата VOD: свежие записи канала + сырые ответы GQL.

Запуск: python -u scripts/diag_chat.py [channel] [client_id]
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from twitch_ai_clipper.twitch_api import COMMENTS_QUERY, TwitchClient  # noqa: E402
from twitch_ai_clipper.utils import setup_logging  # noqa: E402

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
CLIENT_IDS = {
    "web-legacy": "kimne78kx3ncx6brgo4mv6wki5h1ko",
    "android": "kd1unb4b3q4t58fwlpcbzcbnm76a8fp",
    "mobile-tv": "ue6666qo983tsx6so1t0vnawi233wa",
}


def raw_post(client_id: str, query: str, variables: dict, extra_headers: dict | None = None) -> tuple[int, str]:
    headers = {
        "Client-ID": client_id,
        "Content-Type": "application/json",
        "User-Agent": UA,
        "Origin": "https://www.twitch.tv",
        "Referer": "https://www.twitch.tv/",
    }
    if extra_headers:
        headers.update(extra_headers)
    request = urllib.request.Request(
        "https://gql.twitch.tv/gql",
        data=json.dumps({"query": query, "variables": variables}).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")[:500]
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}"


def probe_video(video_id: str, duration: float) -> None:
    print(f"\n--- VOD {video_id} ({duration:.0f}s) ---")
    for name, client_id in CLIENT_IDS.items():
        for offset in (0, int(max(0.0, duration * 0.3))):
            status, body = raw_post(client_id, COMMENTS_QUERY, {"videoID": video_id, "contentOffsetSeconds": offset})
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                print(f"[{name}] offset={offset} HTTP {status} (не JSON): {body[:200]}")
                continue
            video = ((data.get("data") or {}).get("video")) or {}
            comments = video.get("comments") or {}
            edges = comments.get("edges") or []
            errors = data.get("errors") or []
            print(
                f"[{name}] offset={offset} HTTP {status} edges={len(edges)} "
                f"video_null={'video' in data.get('data', {}) and data['data']['video'] is None} "
                f"comments_null={comments == {}}"
                f"{' errors=' + json.dumps(errors, ensure_ascii=False)[:200] if errors else ''}"
            )
            if edges:
                node = edges[0]["node"]
                print("   sample:", json.dumps(node, ensure_ascii=False)[:300])
            time.sleep(0.4)


def main() -> None:
    setup_logging(verbose=False, log_file=ROOT / "probe_out" / "diag_chat.log")
    channel = sys.argv[1] if len(sys.argv) > 1 else "tarik"
    client = TwitchClient(delay=0.2)
    videos = client.list_videos(channel, limit=6)
    print(f"Канал {channel}: найдено {len(videos)} записей")
    for video in videos:
        print(" -", video.get("id"), video.get("lengthSeconds"), (video.get("title") or "")[:60])

    targets = [v for v in videos[:3]]
    targets.append({"id": "2840912067", "lengthSeconds": 19777, "title": "SEN vs. KRÜ (старый тест)"})
    for video in targets:
        probe_video(str(video["id"]), float(video.get("lengthSeconds") or 600))


if __name__ == "__main__":
    main()
