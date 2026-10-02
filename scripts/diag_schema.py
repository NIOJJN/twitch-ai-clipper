"""Интроспекция схемы GQL для типов чата VOD (какие поля реально существуют).

Запуск: python -u scripts/diag_schema.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QUERY = """
query Introspect($names: [String!]!) {
  __schema {
    types { name kind fields { name args { name type { name kind ofType { name } } } type { name kind ofType { name kind ofType { name } } } } }
  }
}
"""

TYPE_QUERY = """
query T($name: String!) {
  __type(name: $name) {
    name
    kind
    fields {
      name
      type { name kind ofType { name kind ofType { name kind ofType { name } } } }
      args { name type { name kind ofType { name kind } } }
    }
  }
}
"""

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"

NAMES = [
    "VideoComment",
    "CommentAuthor",
    "CommentMessage",
    "CommentFragment",
    "EmbeddedEmote",
    "CommentsPageInfo",
    "VideoCommentConnection",
    "VideoCommentEdge",
]


def post(query: str, variables: dict) -> dict:
    request = urllib.request.Request(
        "https://gql.twitch.tv/gql",
        data=json.dumps({"query": query, "variables": variables}).encode("utf-8"),
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


def describe(type_info: dict) -> str:
    def unwrap(node: dict | None) -> str:
        if not node:
            return "?"
        if node.get("name"):
            return node["name"]
        inner = node.get("ofType")
        return f"{node.get('kind', '?')}[{unwrap(inner)}]"

    fields = type_info.get("fields") or []
    return ", ".join(f"{f['name']}: {unwrap(f['type'])}" for f in fields)


def video_comments_args() -> str:
    data = post(TYPE_QUERY, {"name": "Video"})
    fields = ((data.get("data") or {}).get("__type") or {}).get("fields") or []
    for field in fields:
        if field["name"] == "comments":
            args = ", ".join(
                f"{a['name']}: {(a['type'] or {}).get('name') or (a['type'] or {}).get('kind')}"
                for a in (field.get("args") or [])
            )
            return args
    return "(поле Video.comments не найдено)"


def main() -> None:
    print("Video.comments args:", video_comments_args())
    for name in NAMES:
        try:
            data = post(TYPE_QUERY, {"name": name})
        except Exception as exc:  # noqa: BLE001
            print(f"\n{name}: ошибка запроса ({exc})")
            continue
        info = (data.get("data") or {}).get("__type")
        errors = data.get("errors") or []
        if not info:
            print(f"\n{name}: недоступно. errors={json.dumps(errors, ensure_ascii=False)[:200]}")
            continue
        print(f"\n{name} ({info.get('kind')}): {describe(info)[:1500]}")


if __name__ == "__main__":
    main()
