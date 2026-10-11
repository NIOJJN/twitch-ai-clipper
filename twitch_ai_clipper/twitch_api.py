"""Twitch GraphQL: метаданные VOD, список записей канала и ПОЛНОЕ скачивание чата (replay).

Twitch не отдаёт чат VOD через REST v5 (устарел), поэтому используется публичный
GraphQL-эндпоинт ``gql.twitch.tv`` с операцией ``VideoCommentsByOffsetOrCursor``.
Запрос принимает ``contentOffsetSeconds`` — точка привязки; ответ содержит окно
сообщений вокруг этой точки. Чтобы выгрузить чат целиком, мы идём по таймлайну
адаптивным шагом (по последнему offset страницы, с откатом на ``stride`` в тихих
участках) и в конце делаем аудит «дыр».
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from .config import TWITCH_CLIENT_ID, TWITCH_GQL_URL
from .utils import append_jsonl, ensure_dir, fmt_ts, iter_jsonl, write_json

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

VOD_QUERY = """
query VodQuery($id: ID!) {
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

CHANNEL_VIDEOS_QUERY = """
query ChannelVideos($login: String!, $first: Int!) {
  user(login: $login) {
    id
    login
    displayName
    videos(first: $first, type: ARCHIVE, sort: TIME) {
      edges { node { id title lengthSeconds publishedAt createdAt broadcastType status } }
    }
  }
}
"""

COMMENTS_QUERY = """
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
              emote { id emoteID }
            }
          }
        }
      }
      pageInfo { hasNextPage hasPreviousPage }
    }
  }
}
"""

# Поля проверены вживую (introspection у Twitch отключён, поэтому набор валиден
# ровно в таком виде): emoteSetID, userBadges, badges, spam, deleted, channel —
# НЕ существуют в схеме, и любое из них ломает всю операцию (comments -> null).



@dataclass(slots=True)
class ChatMessage:
    """Одно сообщение чата с таймкодом относительно начала VOD."""

    id: str
    offset: float
    user: str
    login: str
    text: str
    emotes: list[str] = field(default_factory=list)
    badges: list[str] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "offset": round(float(self.offset), 3),
            "user": self.user,
            "login": self.login,
            "text": self.text,
            "emotes": self.emotes,
            "badges": self.badges,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ChatMessage":
        return cls(
            id=str(row.get("id", "")),
            offset=float(row.get("offset", 0.0)),
            user=row.get("user", ""),
            login=row.get("login", ""),
            text=row.get("text", ""),
            emotes=list(row.get("emotes") or []),
            badges=list(row.get("badges") or []),
            created_at=row.get("created_at", ""),
        )


@dataclass(slots=True)
class VodInfo:
    """Метаданные VOD."""

    id: str
    title: str
    duration: float
    channel: str
    channel_login: str
    game: str = ""
    created_at: str = ""
    published_at: str = ""
    description: str = ""
    thumbnail: str = ""
    url: str = ""
    local: bool = False  # источник — локальный файл на диске, а не VOD Twitch

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "duration": self.duration,
            "channel": self.channel,
            "channel_login": self.channel_login,
            "game": self.game,
            "created_at": self.created_at,
            "published_at": self.published_at,
            "description": self.description,
            "thumbnail": self.thumbnail,
            "url": self.url or f"https://www.twitch.tv/videos/{self.id}",
            "local": self.local,
        }


class TwitchClient:
    """Клиент публичного Twitch GraphQL (метаданные + чат VOD)."""

    def __init__(
        self,
        client_id: str = TWITCH_CLIENT_ID,
        oauth: str | None = None,
        delay: float = 0.2,
        timeout: float = 30.0,
        max_retries: int = 5,
    ) -> None:
        self.client_id = client_id
        self.oauth = oauth
        self.delay = max(0.0, float(delay))
        self.timeout = timeout
        self.max_retries = max_retries

    # ------------------------------------------------------------------ HTTP
    def _headers(self) -> dict[str, str]:
        headers = {
            "Client-ID": self.client_id,
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            "Origin": "https://www.twitch.tv",
            "Referer": "https://www.twitch.tv/",
        }
        if self.oauth:
            token = self.oauth if self.oauth.lower().startswith("oauth:") else f"oauth:{self.oauth}"
            headers["Authorization"] = token
        return headers

    def _post(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            request = urllib.request.Request(
                TWITCH_GQL_URL, data=payload, headers=self._headers(), method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    if self.delay:
                        time.sleep(self.delay)
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:  # 429/5xx -> backoff
                body = ""
                try:
                    body = exc.read().decode("utf-8", errors="replace")[:300]
                except Exception:  # noqa: BLE001
                    pass
                last_error = RuntimeError(f"HTTP {exc.code}: {body}")
                if exc.code in (429, 500, 502, 503, 504):
                    time.sleep(min(30.0, 1.5 * (2**attempt)))
                    continue
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(min(15.0, 1.0 * (attempt + 1)))
        raise RuntimeError(f"Twitch GQL недоступен: {last_error}")

    def _query(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        result = self._post(query, variables)
        errors = result.get("errors") or []
        if errors:
            message = "; ".join(str(e.get("message")) for e in errors)
            if "integrity" in message.lower():
                raise RuntimeError(f"Twitch GQL: {message}")
            logger.debug("GQL warnings: {}", message)
        return result.get("data") or {}

    # ------------------------------------------------------------- метаданные
    def get_vod(self, video_id: str) -> VodInfo:
        data = self._query(VOD_QUERY, {"id": str(video_id)})
        video = data.get("video") or {}
        if not video:
            raise ValueError(f"VOD {video_id} не найден или недоступен (приватный/удалён?).")
        owner = video.get("owner") or {}
        game = video.get("game") or {}
        return VodInfo(
            id=str(video.get("id") or video_id),
            title=video.get("title") or "",
            duration=float(video.get("lengthSeconds") or 0.0),
            channel=owner.get("displayName") or owner.get("login") or "",
            channel_login=owner.get("login") or "",
            game=game.get("name") or "",
            created_at=video.get("createdAt") or "",
            published_at=video.get("publishedAt") or "",
            description=video.get("description") or "",
            thumbnail=video.get("previewThumbnailURL") or "",
            url=f"https://www.twitch.tv/videos/{video_id}",
        )

    def list_videos(self, login: str, limit: int = 20) -> list[dict[str, Any]]:
        data = self._query(
            CHANNEL_VIDEOS_QUERY, {"login": login.lstrip("@"), "first": int(limit)}
        )
        user = data.get("user") or {}
        edges = ((user.get("videos") or {}).get("edges")) or []
        return [edge.get("node") or {} for edge in edges if edge]

    def resolve_vod_id(self, target: str) -> str:
        """Принимает ID, ссылку на VOD или имя канала (берёт последнюю запись)."""
        target = (target or "").strip()
        if not target:
            raise ValueError("пустой VOD")
        if re.fullmatch(r"\d+", target):
            return target
        match = re.search(r"(?:videos?/)(\d+)", target)
        if match:
            return match.group(1)
        if re.search(r"\d{6,}", target):
            return re.search(r"(\d{6,})", target).group(1)  # type: ignore[union-attr]
        name = re.sub(r"^https?://", "", target)
        name = re.sub(r"^(www\.)?twitch\.tv/", "", name)
        login = name.split("/")[0].split("?")[0].strip()
        if not re.fullmatch(r"[A-Za-z0-9_]{2,30}", login):
            raise ValueError(f"Не удалось разобрать VOD: {target}")
        videos = self.list_videos(login, limit=5)
        if not videos:
            raise ValueError(f"У канала {login} нет доступных записей.")
        logger.info("Канал {}: берём последний VOD {}", login, videos[0].get("id"))
        return str(videos[0]["id"])

    # ------------------------------------------------------------------ чат
    def comments_page(self, video_id: str, offset: float) -> list[ChatMessage]:
        """Одно окно сообщений чата вокруг указанного таймкода VOD."""
        data = self._query(
            COMMENTS_QUERY,
            {"videoID": str(video_id), "contentOffsetSeconds": int(max(0.0, offset))},
        )
        video = data.get("video") or {}
        comments = video.get("comments") or {}
        edges = comments.get("edges") or []
        messages: list[ChatMessage] = []
        for edge in edges:
            node = (edge or {}).get("node") or {}
            if not node:
                continue
            commenter = node.get("commenter") or {}
            fragments = ((node.get("message") or {}).get("fragments")) or []
            text_parts: list[str] = []
            emotes: list[str] = []
            for fragment in fragments:
                if not isinstance(fragment, dict):
                    continue
                text_parts.append(fragment.get("text") or "")
                emote = fragment.get("emote") or {}
                emote_id = emote.get("emoteID") or emote.get("id")
                if emote_id:
                    emotes.append(str(emote_id))
            badges = [
                str(b.get("setID") or b.get("id"))
                for b in (node.get("userBadges") or [])
                if isinstance(b, dict)
            ]
            messages.append(
                ChatMessage(
                    id=str(node.get("id") or ""),
                    offset=float(node.get("contentOffsetSeconds") or 0.0),
                    user=commenter.get("displayName") or commenter.get("login") or "",
                    login=commenter.get("login") or "",
                    text="".join(text_parts).strip(),
                    emotes=emotes,
                    badges=badges,
                    created_at=node.get("createdAt") or "",
                )
            )
        return messages

    @staticmethod
    def _next_offset(
        page: list[ChatMessage], offset: float, stride: float, backward_streak: int
    ) -> tuple[float, int]:
        """Куда идти дальше: за последним сообщением окна либо шаг вперёд."""
        if not page:
            return offset + stride, 0
        max_offset = max(m.offset for m in page)
        if max_offset > offset:
            return max_offset + 1.0, 0
        backward_streak += 1
        return offset + (5.0 if backward_streak <= 4 else stride), backward_streak

    def download_chat(
        self,
        video_id: str,
        out_path: Path | str,
        duration: float | None = None,
        stride: float = 45.0,
        resume: bool = True,
        gap_audit: bool = True,
        min_gap: float = 240.0,
        max_requests: int = 0,
        heartbeat: float = 60.0,
    ) -> dict[str, Any]:
        """Скачивает чат VOD целиком в JSONL. Возвращает статистику прогона.

        Прогресс пишется в лог не чаще, чем раз в ``heartbeat`` секунд прогресса,
        чтобы длинные стримы (5-10 ч) не заспамили консоль.
        """
        out_path = Path(out_path)
        ensure_dir(out_path.parent)
        seen: dict[str, ChatMessage] = {}
        if resume and out_path.exists():
            for row in iter_jsonl(out_path):
                message = ChatMessage.from_dict(row)
                if message.id:
                    seen[message.id] = message
        total = float(duration) if duration else None
        if seen:
            logger.info(
                "Чат: продолжаем с {} (уже {} сообщений, до {})",
                out_path.name,
                len(seen),
                fmt_ts(max(m.offset for m in seen.values())),
            )
            offset = max(m.offset for m in seen.values()) + 1.0
        else:
            offset = 0.0

        buffer: list[ChatMessage] = []
        requests = 0
        backward_streak = 0
        next_report = heartbeat
        while True:
            if total is not None and offset >= total:
                break
            if max_requests and requests >= max_requests:
                logger.info("Чат: лимит запросов ({}), останавливаемся.", max_requests)
                break
            page = self.comments_page(video_id, offset)
            requests += 1
            fresh = [m for m in page if m.id and m.id not in seen]
            for message in fresh:
                seen[message.id] = message
            buffer.extend(fresh)
            if len(buffer) >= 300:
                append_jsonl(out_path, [m.to_dict() for m in buffer])
                buffer = []
            if offset >= next_report:
                percent = f"{offset / total * 100:5.1f}%" if total else "  ?  "
                logger.info(
                    "Чат: {} {} | сообщений {} | запросов {}",
                    percent,
                    fmt_ts(offset),
                    len(seen),
                    requests,
                )
                next_report = offset + heartbeat
            offset, backward_streak = self._next_offset(page, offset, stride, backward_streak)
        if buffer:
            append_jsonl(out_path, [m.to_dict() for m in buffer])

        stats: dict[str, Any] = {
            "video_id": str(video_id),
            "messages": len(seen),
            "requests": requests,
            "duration": total,
        }
        if gap_audit and seen and (total is None or len(seen) > 0):
            inserted = self._gap_audit(video_id, seen, out_path, total, min_gap, max_requests)
            stats["gap_audit_inserted"] = inserted
            stats["messages"] = len(seen)
        if seen:
            offsets = [m.offset for m in seen.values()]
            stats["first_offset"] = min(offsets)
            stats["last_offset"] = max(offsets)
            stats["unique_users"] = len({m.login for m in seen.values() if m.login})
        write_json(out_path.with_suffix(".stats.json"), stats)
        logger.info(
            "Чат готов: {} сообщений, {} запросов -> {}",
            stats["messages"],
            stats["requests"],
            out_path,
        )
        return stats



    def _gap_audit(
        self,
        video_id: str,
        seen: dict[str, ChatMessage],
        out_path: Path,
        total: float | None,
        min_gap: float,
        max_requests: int,
        max_rounds: int = 4,
    ) -> int:
        """Ищет «дыры» в таймлайне чата и добирает сообщения из них.

        Ответ GQL не всегда покрывает таймлайн равномерно, поэтому после основного
        прохода проверяем разрывы > ``min_gap`` секунд: если внутри есть сообщения,
        они попадут в файл.
        """
        inserted_total = 0
        for round_index in range(max_rounds):
            offsets = sorted(m.offset for m in seen.values())
            gaps = [(a, b) for a, b in zip(offsets, offsets[1:]) if b - a > min_gap]
            if total is not None and offsets and total - offsets[-1] > min_gap:
                gaps.append((offsets[-1], total))
            if not gaps:
                break
            inserted = 0
            for start, end in gaps:
                span = end - start
                probes = (start + span / 2, start + min(120.0, span / 4), start + span * 0.75)
                for probe in probes:
                    page = self.comments_page(video_id, probe)
                    fresh = [
                        m for m in page if m.id and m.id not in seen and start <= m.offset <= end
                    ]
                    if fresh:
                        for message in fresh:
                            seen[message.id] = message
                        append_jsonl(out_path, [m.to_dict() for m in fresh])
                        inserted += len(fresh)
                        break
            inserted_total += inserted
            logger.debug(
                "Аудит чата #{}: разрывов {}, добрано {}",
                round_index + 1,
                len(gaps),
                inserted,
            )
            if not inserted:
                break
        if inserted_total:
            logger.info("Аудит чата: добрано {} сообщений из разрывов", inserted_total)
        return inserted_total

