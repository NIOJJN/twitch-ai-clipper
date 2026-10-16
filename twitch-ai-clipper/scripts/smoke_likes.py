"""Smoke-тест лайков: API галереи + генерация index.html."""
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, r"d:\twitch-ai-clipper")

from fastapi.testclient import TestClient  # noqa: E402

from twitch_ai_clipper.webui import build_app, find_vods  # noqa: E402


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    vod = tmp / "12345"
    vod.mkdir()
    (vod / "metadata.json").write_text(
        json.dumps({"title": "Test", "channel": "ch", "duration": 100}), encoding="utf-8"
    )
    (vod / "moments.json").write_text(
        json.dumps([{"index": 0, "start": 1.0}]), encoding="utf-8"
    )

    client = TestClient(build_app(tmp))

    # пустое хранилище
    r = client.get("/api/vods/12345/likes")
    assert r.status_code == 200 and r.json() == {}, r.text

    # поставить / снять / ограничение снизу
    assert client.post("/api/vods/12345/likes/0?action=up").json()["likes"] == 1
    assert client.post("/api/vods/12345/likes/0?action=up").json()["likes"] == 2
    assert client.post("/api/vods/12345/likes/0?action=down").json()["likes"] == 1
    assert client.post("/api/vods/12345/likes/0?action=down").json()["likes"] == 0
    assert client.post("/api/vods/12345/likes/0?action=down").json()["likes"] == 0

    # неизвестный VOD
    assert client.post("/api/vods/99999/likes/0").status_code == 404
    assert client.get("/api/vods/99999/likes").status_code == 404

    # персистентность на диске
    likes_file = json.loads((vod / "likes.json").read_text(encoding="utf-8"))
    assert likes_file == {"0": 0}, likes_file

    # лендинг: сумма лайков
    assert client.post("/api/vods/12345/likes/3?action=up").status_code == 200
    vods = find_vods(tmp)
    assert vods[0]["likes"] == 1, vods
    landing = client.get("/").text
    assert re.search(r"\u043b\u0430\u0439\u043a\u043e\u0432: 1", landing), "нет счётчика на лендинге"

    print("API smoke: OK")


if __name__ == "__main__":
    main()
