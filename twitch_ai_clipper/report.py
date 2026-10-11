"""Отчёты: moments.json, report.md и локальная HTML-галерея клипов."""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Sequence

from .config import Settings
from .detect import Moment
from .modes import chapter_lines, get_mode
from .twitch_api import VodInfo
from .utils import ensure_dir, fmt_duration, write_json

CATEGORY_EMOJI = {
    "clutch": "🎯",
    "skill": "🔥",
    "funny": "😂",
    "fail": "🤡",
    "rage": "😡",
    "reaction": "😱",
    "story": "📖",
    "drama": "🍿",
    "hype": "🚀",
    "other": "⭐",
}


def render_markdown(
    vod: VodInfo,
    moments: Sequence[Moment],
    chat_summary: dict[str, Any] | None,
    settings: Settings,
    ai_summary: str = "",
    vod_url: str | None = None,
) -> str:
    """Собирает читаемый отчёт в Markdown."""
    url = vod_url or vod.url
    mode = get_mode(settings.mode)
    is_local = getattr(vod, "local", False)
    lines: list[str] = []
    lines.append(f"# Хайлайты: {vod.title or vod.id}")
    lines.append("")
    lines.append(f"- **Канал:** {vod.channel or '?'} ({vod.channel_login or '?'})")
    if vod.game:
        lines.append(f"- **Игра/категория:** {vod.game}")
    lines.append(f"- **Длительность:** {fmt_duration(vod.duration)} ({mode.stamp(vod.duration)})")
    lines.append(f"- **VOD:** {url}")
    if vod.created_at:
        lines.append(f"- **Дата:** {vod.created_at}")
    lines.append(f"- **Найдено моментов:** {len(moments)}")
    lines.append(f"- **Мод:** {mode.title} (`{mode.name}`)")
    lines.append(f"- **Модель ASR:** {settings.whisper_model} (язык: {settings.language or 'авто'})")
    lines.append(f"- **LLM:** {settings.llm_provider} / {settings.llm_model}")
    if settings.limit_seconds:
        lines.append(f"- **Ограничение анализа:** первые {fmt_ts(settings.limit_seconds)}")
    lines.append("")
    if chat_summary:
        lines.append("## Чат")
        lines.append("")
        lines.append(f"- Сообщений: {chat_summary.get('total_messages', 0)}")
        lines.append(f"- Уникальных зрителей: {chat_summary.get('unique_users', 0)}")
        emotes = chat_summary.get("emotes") or {}
        if emotes:
            top = ", ".join(f"{name} ×{count}" for name, count in list(emotes.items())[:10])
            lines.append(f"- Топ эмоутов: {top}")
        lines.append("")
    if ai_summary:
        lines.append("## Резюме стрима (AI)")
        lines.append("")
        lines.append(ai_summary)
        lines.append("")
    lines.append("## Моменты")
    lines.append("")
    if mode.chapter_list:
        # Оглавление ровно в том виде, в котором его ждёт мод (2:04:19 - Название).
        lines.append(f"### Таймкоды ({mode.title})")
        lines.append("")
        lines.append("```")
        lines.extend(moment.chapter(mode.name) for moment in moments)
        lines.append("```")
        lines.append("")
    lines.append("| # | Таймкод | Ссылка | Тип | Score | Название | Источник |")
    lines.append("|---|---------|--------|-----|-------|----------|----------|")
    for moment in moments:
        emoji = CATEGORY_EMOJI.get(moment.category, "⭐")
        stamp = moment.stamp(mode.name)
        clip = Path(moment.clip_path).name if moment.clip_path else ""
        title = moment.title.replace("|", "/")
        if is_local:
            # Ссылка «?t=» на Twitch для локального файла бессмысленна — показываем таймкод.
            clip_cell = stamp + (f" · {clip}" if clip else "")
        else:
            deep_link = f"{url}?t={int(moment.start)}s"
            clip_cell = f"[{stamp}]({deep_link})" + (f" · {clip}" if clip else "")
        origin = "речь" if moment.transcript.strip() else "чат/звук"
        lines.append(
            f"| {moment.index} | `{stamp}` | {clip_cell} | {emoji} {moment.category} "
            f"| {moment.score:.0f} | {title} | {origin} |"
        )
    lines.append("")
    lines.append(
        "**Источник** — откуда взят заголовок: «речь» = распознанная реплика стримера, "
        "«чат/звук» = в этом окне не было речи, заголовок собран из реакции чата."
    )
    lines.append("")
    for moment in moments:
        lines.append(f"### {moment.index}. {moment.title} ({moment.stamp(mode.name)})")
        lines.append("")
        lines.append(f"- Тип: {moment.category} · score {moment.score:.0f} · источник: {moment.source}")
        lines.append(
            f"- Интервал: {mode.stamp(moment.start)} → {mode.stamp(moment.end)} "
            f"({moment.duration:.0f} с)"
        )
        if moment.reason:
            lines.append(f"- Почему: {moment.reason}")
        if moment.clip_path:
            lines.append(f"- Файл: `{moment.clip_path}`")
        if moment.thumbnail_path:
            lines.append(f"- Превью: `{moment.thumbnail_path}`")
        if moment.transcript:
            lines.append("")
            lines.append("**Расшифровка:**")
            lines.append("")
            lines.append("```")
            lines.append(moment.transcript[:1500])
            lines.append("```")
        if moment.chat_excerpt:
            lines.append("")
            lines.append("**Чат:**")
            lines.append("")
            lines.append("```")
            lines.extend(moment.chat_excerpt[:15])
            lines.append("```")
        lines.append("")
    return "\n".join(lines)



def _relative_media(path: str, out_dir: Path | str) -> str:
    """Путь к файлу относительно папки VOD — чтобы index.html работал и с диска, и через serve."""
    if not path:
        return ""
    try:
        relative = Path(path).resolve().relative_to(Path(out_dir).resolve())
        return relative.as_posix()
    except (ValueError, OSError):
        return Path(path).name


def render_html(
    vod: VodInfo,
    moments: Sequence[Moment],
    settings: Settings,
    ai_summary: str = "",
    vod_url: str | None = None,
    out_dir: Path | str = ".",
) -> str:
    """Простая галерея: плееры клипов, таймкоды, расшифровка (локальный просмотр)."""
    url = vod_url or vod.url
    mode = get_mode(settings.mode)
    is_local = getattr(vod, "local", False)
    cards: list[str] = []
    for moment in moments:
        emoji = CATEGORY_EMOJI.get(moment.category, "⭐")
        clip = _relative_media(moment.clip_path, out_dir)
        thumb = _relative_media(moment.thumbnail_path, out_dir)
        media = ""
        if clip:
            poster = f' poster="{html.escape(thumb)}"' if thumb else ""
            media = f'<video controls preload="metadata"{poster} src="{html.escape(clip)}"></video>'
        elif thumb:
            media = f'<img class="poster" src="{html.escape(thumb)}" alt="превью">'
        elif not is_local:
            # Клип ещё не нарезан — показываем таймкод и ссылку на сам Twitch.
            link = f"{url}?t={int(moment.start)}s"
            media = (
                f'<div class="pending">Клип не нарезан &middot; '
                f'<a href="{html.escape(link)}" target="_blank">смотреть на Twitch с '
                f'{moment.stamp(mode.name)}</a></div>'
            )
        else:
            media = f'<div class="pending">Клип не нарезан &middot; {moment.stamp(mode.name)}</div>'
        chat_html = html.escape("\n".join(moment.chat_excerpt[:12]))
        transcript_html = html.escape(moment.transcript[:900])
        origin = (
            '<span class="tag speech">речь стримера</span>'
            if moment.transcript.strip()
            else '<span class="tag chat">нет речи — из звука</span>' if is_local
            else '<span class="tag chat">нет речи — из чата</span>'
        )
        stamp_html = (
            moment.stamp(mode.name) if is_local
            else f'<a href="{html.escape(url)}?t={int(moment.start)}s" target="_blank">{moment.stamp(mode.name)}</a>'
        )
        cards.append(
            f"""<article class="card">
  <header>
    <span class="emoji">{emoji}</span>
    <h2>{html.escape(moment.title)}</h2>
    <span class="score">{moment.score:.0f}</span>
  </header>
  <div class="meta">
    {stamp_html}
    · {html.escape(moment.category)} · {moment.duration:.0f} с · {html.escape(moment.source)}
    · {origin}
  </div>
  {media}
  <p class="reason">{html.escape(moment.reason)}</p>
  <details><summary>Расшифровка</summary><pre>{transcript_html}</pre></details>
  <details><summary>Чат</summary><pre>{chat_html}</pre></details>
</article>"""
        )
    summary_block = f'<p class="reason">{html.escape(ai_summary)}</p>' if ai_summary else ""
    body = "".join(cards) if cards else '<p class="meta">Моменты не найдены.</p>'
    chapters_block = ""
    if mode.chapter_list and moments:
        # Оглавление таймкодами в оформлении мода, каждая строка — ссылка на VOD.
        if is_local:
            items = "\n".join(html.escape(moment.chapter(mode.name)) for moment in moments)
        else:
            items = "".join(
                f'<a href="{html.escape(url)}?t={int(moment.start)}s" target="_blank">'
                f"{html.escape(moment.chapter(mode.name))}</a>"
                for moment in moments
            )
        chapters_block = (
            f'<nav class="chapters"><div class="chapters-title">Таймкоды — {html.escape(mode.title)}</div>'
            f"<pre>{items}</pre></nav>"
        )
    source_html = (
        f"Файл: {html.escape(url)}" if is_local
        else f'<a href="{html.escape(url)}" target="_blank">{html.escape(url)}</a>'
    )
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<title>Хайлайты: {html.escape(vod.title or vod.id)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ color-scheme: dark; }}
body {{ margin: 0; font-family: "Segoe UI", system-ui, sans-serif; background: #0e0e10; color: #efeff1; }}
header.top {{ padding: 24px; background: linear-gradient(135deg, #9147ff22, #0e0e1000); border-bottom: 1px solid #2a2a2d; }}
h1 {{ margin: 0 0 8px; font-size: 22px; }}
.meta {{ color: #adadb8; font-size: 14px; margin-top: 6px; }}
.wrap {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 18px; padding: 20px; }}
.card {{ background: #18181b; border: 1px solid #2a2a2d; border-radius: 12px; padding: 14px; }}
.card header {{ display: flex; align-items: center; gap: 8px; }}
.card h2 {{ font-size: 16px; margin: 0; flex: 1; }}
.score {{ background: #9147ff; border-radius: 999px; padding: 2px 10px; font-size: 13px; }}
video {{ width: 100%; border-radius: 8px; margin: 10px 0; background: #000; }}
.poster {{ width: 100%; border-radius: 8px; margin: 10px 0; display: block; }}
.pending {{ margin: 10px 0; padding: 12px; border: 1px dashed #3a3a3d; border-radius: 8px; font-size: 13px; color: #adadb8; }}
.tag {{ display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 11px; }}
.tag.speech {{ background: #103d24; color: #6ee7a0; }}
.tag.chat {{ background: #3a3320; color: #e7c86e; }}
a {{ color: #a970ff; }}
.reason {{ color: #d3d3d8; font-size: 14px; }}
pre {{ white-space: pre-wrap; font-size: 12px; color: #adadb8; max-height: 220px; overflow: auto; }}
details summary {{ cursor: pointer; color: #a970ff; font-size: 13px; margin-top: 6px; }}
.chapters {{ margin-top: 14px; }}
.chapters-title {{ color: #adadb8; font-size: 13px; margin-bottom: 4px; }}
.chapters pre {{ max-height: 320px; }}
.chapters pre a {{ display: block; text-decoration: none; padding: 1px 0; }}
.chapters pre a:hover {{ color: #fff; text-decoration: underline; }}
</style></head>
<body>
<header class="top">
  <h1>{html.escape(vod.title or vod.id)}</h1>
  <div class="meta">{html.escape(vod.channel)} · {html.escape(vod.game)} · {fmt_duration(vod.duration)}
  · моментов: {len(moments)} · модель {html.escape(settings.whisper_model)}
  · мод: {html.escape(mode.title)}</div>
  <div class="meta">{source_html}</div>
  {summary_block}
  {chapters_block}
</header>
<main class="wrap">
{body}
</main>
</body></html>"""


def write_reports(
    out_dir: Path | str,
    vod: VodInfo,
    moments: Sequence[Moment],
    settings: Settings,
    chat_summary: dict[str, Any] | None = None,
    ai_summary: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Пишет moments.json, report.md, index.html (и run_summary.json). Возвращает пути.

    В модах с оглавлением (например ``toparachevt``) дополнительно пишется
    ``timecodes.txt`` — готовый список глав построчно: ``2:04:19 - Любимый цвет``.
    """
    out_dir = ensure_dir(out_dir)
    mode = get_mode(settings.mode)
    moments_path = write_json(out_dir / "moments.json", [m.to_dict(mode.name) for m in moments])
    report_path = out_dir / "report.md"
    report_path.write_text(
        render_markdown(vod, moments, chat_summary, settings, ai_summary), encoding="utf-8"
    )
    html_path = out_dir / "index.html"
    html_path.write_text(
        render_html(vod, moments, settings, ai_summary, out_dir=out_dir), encoding="utf-8"
    )
    paths = {"moments": str(moments_path), "report": str(report_path), "html": str(html_path)}
    if mode.chapter_list and moments:
        timecodes_path = out_dir / "timecodes.txt"
        text = "\n".join(chapter_lines([(m.start, m.title) for m in moments], mode.name)) + "\n"
        timecodes_path.write_text(text, encoding="utf-8")
        paths["timecodes"] = str(timecodes_path)
    if extra:
        write_json(out_dir / "run_summary.json", extra)
    return paths
