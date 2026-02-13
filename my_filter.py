"""Sample user filter for `evaluate-broadcast-events`.

- `before_evaluate_events`: optional setup hook run once before evaluation starts.
- `evaluate_event`: matches events whose title contains 🈟 or 🈡.
- `handle_matched_event`: accumulates matched event text.
- `after_evaluate_events`: posts accumulated messages to Slack Incoming Webhook.
"""

from __future__ import annotations

import os

import aiohttp


SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
_MATCHED_TEXTS: list[str] = []


async def before_evaluate_events() -> None:
    _MATCHED_TEXTS.clear()


async def evaluate_event(metadata: dict) -> bool:
    title = (metadata.get("metadata_title") or metadata.get("title") or "").strip()
    return "🈟" in title or "🈡" in title


async def handle_matched_event(program: dict) -> None:
    title = (program.get("metadata_title") or program.get("title") or "(no title)").strip()
    channel = (program.get("channel_name") or "unknown channel").strip()
    start_at = program.get("li_start_at") or "unknown time"
    detail = (program.get("detail") or program.get("metadata_detail") or "").strip()
    event_url = program.get("event_url") or ""

    text = "\n".join(
        line
        for line in [
            f"番組マッチ: {title}",
            f"チャンネル: {channel}",
            f"開始: {start_at}",
            f"概要: {detail}" if detail else "",
            event_url,
        ]
        if line
    )
    _MATCHED_TEXTS.append(text)


async def after_evaluate_events() -> None:
    if not SLACK_WEBHOOK_URL or not _MATCHED_TEXTS:
        return

    payload = {"text": "\n\n".join(_MATCHED_TEXTS)}

    async with aiohttp.ClientSession() as session:
        async with session.post(SLACK_WEBHOOK_URL, json=payload) as response:
            response.raise_for_status()
