"""Sample user filter for `evaluate-broadcast-events`.

- `evaluate_event`: matches events whose title contains 🈟 or 🈡.
- `handle_matched_event`: posts matched event info to Slack Incoming Webhook via aiohttp.
"""

from __future__ import annotations

import os

import aiohttp


SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")


async def evaluate_event(metadata: dict) -> bool:
    title = (metadata.get("metadata_title") or metadata.get("title") or "").strip()
    return "🈟" in title or "🈡" in title


async def handle_matched_event(program: dict) -> None:
    if not SLACK_WEBHOOK_URL:
        return

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

    payload = {"text": text}

    async with aiohttp.ClientSession() as session:
        async with session.post(SLACK_WEBHOOK_URL, json=payload) as response:
            response.raise_for_status()
