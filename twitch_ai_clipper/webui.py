"""Локальная галерея результатов: FastAPI + статика папки output/.

Запуск: python -m twitch_ai_clipper serve --port 8765
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import read_json

LANDING = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Twitch AI Clipper</title>
<style>
body {{ margin:0; font-family:"Segoe UI",system-ui,sans-serif; background:#0e0e10; color:#efeff1; }}
h1 {{ padding:24px 24px 0; font-size:22px; }}
ul {{ list-style:none; padding:0 24px 40px; max-width:1100px; }}
li {{ background:#18181b; border:1px solid #2a2a2d; border-radius:10px; padding:14px 16px; margin-bottom:12px; }}
a {{ color:#a970ff; text-decoration:none; }}
.meta {{ color:#adadb8; font-size:13px; margin-top:6px; }}
</style></head><body>
<h1>Результаты анализа VOD</h1>
<ul>{items}</ul>
</body></html>"""


def find_vods(output_dir: Path | str) -> list[dict[str, Any]]:
    """Сканирует папку результатов и собирает список VOD."""
    root = Path(output_dir)
    found: list[dict[str, Any]] = []
    if not root.exists():
        return found
    for child in sorted(root.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        meta = read_json(child / "metadata.json", {}) or {}
        moments = read_json(child / "moments.json", []) or []
        if not meta and not moments:
            continue
        clips = len(list((child / "clips").glob("*.mp4"))) if (child / "clips").exists() else 0
        found.append(
            {
                "id": child.name,
                "title": meta.get("title", child.name),
                "channel": meta.get("channel", ""),
                "duration": meta.get("duration", 0),
                "moments": len(moments),
                "clips": clips,
                "has_report": (child / "report.md").exists(),
            }
        )
    return found


def build_app(output_dir: Path | str) -> Any:
    """Создаёт FastAPI-приложение с API и статикой."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    root = Path(output_dir).resolve()
    app = FastAPI(title="Twitch AI Clipper", version="1.0.0")

    @app.get("/api/vods")
    def api_vods() -> Any:
        return JSONResponse(find_vods(root))

    @app.get("/api/vods/{vod_id}/moments")
    def api_moments(vod_id: str) -> Any:
        return JSONResponse(read_json(root / vod_id / "moments.json", []) or [])

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        items = []
        for vod in find_vods(root):
            items.append(
                "<li>"
                f'<a href="/vod/{vod["id"]}/index.html">{vod["title"] or vod["id"]}</a>'
                f'<div class="meta">{vod["channel"]} · моментов: {vod["moments"]} · '
                f'клипов: {vod["clips"]} · '
                f'<a href="/vod/{vod["id"]}/report.md">report.md</a> · '
                f'<a href="/api/vods/{vod["id"]}/moments">JSON</a></div>'
                "</li>"
            )
        return LANDING.format(items="".join(items) or '<li class="meta">Пока ничего нет.</li>')

    if root.exists():
        app.mount("/vod", StaticFiles(directory=str(root), html=True), name="vod")
    else:
        root.mkdir(parents=True, exist_ok=True)
        app.mount("/vod", StaticFiles(directory=str(root), html=True), name="vod")
    return app


def serve(output_dir: Path | str, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Поднимает локальный сервер галереи."""
    import uvicorn

    print(f"Галерея: http://{host}:{port}/  (Ctrl+C — остановить)")
    uvicorn.run(build_app(output_dir), host=host, port=int(port), log_level="info")
