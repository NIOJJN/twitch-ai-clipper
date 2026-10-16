"""Подбор валидных полей для VideoComment: какие поля схема реально принимает.

Запуск: python -u scripts/diag_fields.py [video_id]
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

TEMPLATE = """
query VideoCommentsByOffsetOrCursor($videoID: ID!, $contentOffsetSeconds: Int) {
  video(id: $videoID) {
    comments(contentOffsetSeconds: $contentOffsetSeconds) {
      edges {
        cursor
        node {
          id
          contentOffsetSeconds
          createdAt
          commenter { id login displayName }
          %s
        }
      }
      pageInfo { hasNextPage hasPreviousPage }
    }
  }
}
"""

CANDIDATES: dict[str, str] = {
    "message.fragments": "message { fragments { text emote { id emoteID } } }",
    "message.fragments+emoteSetID": "message { fragments { text emote { id emoteID emoteSetID } } }",
    "userBadges": "userBadges { id setID version }",
    "badges": "badges { id setID version }",
    "commenter+displayName+id": "commenter { id login displayName }",
    "message.userColor": "message { userColor }",
    "spam": "spam { isSpam }",
    "replies": "replies { edges { node { id } } }",
    "deleted": "deleted",
    "channel": "channel { id login displayName }",
    "message.fragments.full": (
        "message { fragments { text emote { id emoteID emoteSetID } } userColor }"
    ),
}


def post(query: str, video_id: str, offset: int) -> dict:
    request = urllib.request.Request(
        "https://gql.twitch.tv/gql",
        data=json.dumps({"query": query, "variables": {"videoID": video_id, "contentOffsetSeconds": offset}}).encode("utf-8"),
        headers={
            "Client-ID": CLIENT_ID,
            "Content-Type": "application/json",
            "User-Agent": UA,
            "Origin": "https://www.twitch.tv",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    video_id = sys.argv[1] if len(sys.argv) > 1 else "2840912067"
    for name, snippet in CANDIDATES.items():
        try:
            data = post(TEMPLATE % snippet, video_id, 200)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:34s} -> ошибка запроса: {exc}")
            continue
        errors = data.get("errors") or []
        edges = (((data.get("data") or {}).get("video") or {}).get("comments") or {}).get("edges") or []
        if errors:
            print(f"{name:34s} -> ОШИБКА: {errors[0].get('message')[:110]}")
        elif not edges:
            print(f"{name:34s} -> OK, но 0 сообщений в окне")
        else:
            node = edges[0].get("node") or {}
            keys = sorted(node.keys())
            print(f"{name:34s} -> OK, {len(edges)} сообщений, поля: {keys}")
            for key in ("message", "userBadges", "badges"):
                if key in node:
                    print(f"      {key}: {json.dumps(node[key], ensure_ascii=False)[:220]}")
        time.sleep(0.4)


if __name__ == "__main__":
    main()
