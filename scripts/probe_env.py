"""Разведка окружения и проверка допущений по Twitch GQL / faster-whisper / Ollama.

Запуск:  python scripts/probe_env.py [video_id]
Результат: probe_out/*.json и сводка в консоль.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"
GQL_URL = "https://gql.twitch.tv/gql"
OUT = Path(__file__).resolve().parent.parent / "probe_out"

COMMENTS_Q = """
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
          message {
            fragments {
              text
              emote { id emoteID emoteSetID }
            }
          }
          userBadges { id setID version }
        }
      }
      pageInfo { hasNextPage hasPreviousPage }
    }
  }
}
"""

META_Q = """
query VodMeta($id: ID!) {
  video(id: $id) {
    id
    title
    lengthSeconds
    createdAt
    publishedAt
    broadcastType
    status
    previewThumbnailURL
    seekPreviewsURL
    description
    owner { login displayName }
    game { name }
  }
}
"""


def gql(query: str, variables: dict, retries: int = 3) -> dict:
    payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(
            GQL_URL,
            data=payload,
            headers={
                "Client-ID": CLIENT_ID,
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GQL failed: {last}")


def fetch_page(video_id: str, offset: int) -> list[dict]:
    res = gql(COMMENTS_Q, {"videoID": str(video_id), "contentOffsetSeconds": int(offset)})
    video = (res.get("data") or {}).get("video") or {}
    comments = video.get("comments") or {}
    edges = comments.get("edges") or []
    return [e["node"] for e in edges if e.get("node")]


def probe_comments(video_id: str, window_end: int = 600) -> None:
    print("\n=== 1. Twitch GQL comments ===")
    page = fetch_page(video_id, 0)
    print(f"edges: {len(page)}")
    if page:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "comments_page0.json").write_text(
            json.dumps(page[:10], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("sample node:", json.dumps(page[0], ensure_ascii=False)[:600])
        print(
            "offsets:",
            [n["contentOffsetSeconds"] for n in page[:5]],
            "...",
            [n["contentOffsetSeconds"] for n in page[-3:]],
        )

    seen_a: dict[str, dict] = {}
    offset, req = 0, 0
    while offset < window_end and req < 400:
        nodes = fetch_page(video_id, offset)
        req += 1
        for n in nodes:
            seen_a.setdefault(n["id"], n)
        if nodes:
            offset = max(n["contentOffsetSeconds"] for n in nodes) + 1
        else:
            offset += 45
        time.sleep(0.15)
    print(f"adaptive: requests={req} unique_comments={len(seen_a)}")

    seen_b: dict[str, dict] = {}
    req = 0
    for offset in range(0, window_end, 5):
        nodes = fetch_page(video_id, offset)
        req += 1
        for n in nodes:
            seen_b.setdefault(n["id"], n)
        time.sleep(0.15)
    print(f"stride=5: requests={req} unique_comments={len(seen_b)}")
    missing = set(seen_b) - set(seen_a)
    print(f"adaptive misses vs stride5: {len(missing)}")
    if missing:
        print(
            "missed offsets sample:",
            sorted(seen_b[i]["contentOffsetSeconds"] for i in missing)[:10],
        )


def probe_whisper() -> None:
    print("\n=== 2. faster-whisper / CUDA ===")
    try:
        import numpy as np
        import soundfile as sf
        from faster_whisper import WhisperModel
    except Exception as exc:  # noqa: BLE001
        print("import failed:", exc)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    wav = OUT / "silence.wav"
    sf.write(str(wav), np.zeros(16000 * 8, dtype="float32"), 16000)
    for device, compute in (("cuda", "float16"), ("cpu", "int8")):
        t0 = time.time()
        try:
            model = WhisperModel("tiny", device=device, compute_type=compute)
            segs, info = model.transcribe(str(wav), vad_filter=True)
            text = " ".join(s.text for s in segs)
            print(f"{device}/{compute}: OK ({time.time() - t0:.1f}s) lang={info.language} text={text!r}")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"{device}/{compute}: FAILED -> {type(exc).__name__}: {str(exc)[:300]}")


def probe_ollama() -> None:
    print("\n=== 3. Ollama ===")
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        print("available models:", [m.get("name") for m in data.get("models", [])])
    except Exception as exc:  # noqa: BLE001
        print("not reachable:", exc)


def probe_binaries() -> None:
    print("\n=== 4. Бинарники / версии ===")
    for cmd in (["ffmpeg", "-version"], [sys.executable, "-m", "yt_dlp", "--version"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            print(" ".join(cmd), "->", (out.stdout or out.stderr).splitlines()[0][:120])
        except Exception as exc:  # noqa: BLE001
            print(" ".join(cmd), "-> FAILED", exc)
    import importlib

    for mod in ("faster_whisper", "ctranslate2", "torch", "yt_dlp", "numpy",
                "soundfile", "loguru", "fastapi", "uvicorn"):
        try:
            m = importlib.import_module(mod)
            print(f"{mod}: {getattr(m, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001
            print(f"{mod}: missing ({exc})")
    try:
        import torch

        print("torch.cuda.available:", torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    video_id = sys.argv[1] if len(sys.argv) > 1 else "2840912067"
    print("meta:", json.dumps(gql(META_Q, {"id": video_id}), ensure_ascii=False)[:600])
    probe_comments(video_id)
    probe_whisper()
    probe_ollama()
    probe_binaries()


if __name__ == "__main__":
    main()
