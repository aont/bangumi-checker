#!/usr/bin/env python3
import argparse
import asyncio
import datetime
import json
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin, urlparse

import aiohttp
import aiosqlite
from lxml import html

BASE_URL = "https://bangumi.org"
TERRESTRIAL_GROUPS = {
    37: "Saitama",
    40: "Chiba",
    42: "Tokyo",
    45: "Kanagawa",
}


@dataclass
class SourceRequest:
    source_type: str
    url: str
    broadcast_date: str
    ggm_group_id: Optional[int] = None


async def fetch_html(session: aiohttp.ClientSession, req: SourceRequest) -> str:
    async with session.get(req.url) as response:
        response.raise_for_status()
        return await response.text()


def parse_channel_names(tree: html.HtmlElement) -> list[str]:
    channels = [
        " ".join(node.cssselect("p")[0].itertext()).strip()
        for node in tree.cssselect("#contents > #ch_area > ul > li.js_channel.topmost")
        if node.cssselect("p")
    ]
    return channels


def parse_event_rows(req: SourceRequest, raw_html: str) -> list[dict]:
    tree = html.fromstring(raw_html)
    channels = parse_channel_names(tree)
    rows: list[dict] = []

    for channel_ul in tree.cssselect("#contents > #program_area > ul"):
        ul_id = channel_ul.get("id", "")
        if not ul_id.startswith("program_line_"):
            continue

        try:
            channel_index = int(ul_id.rsplit("_", 1)[1])
        except ValueError:
            continue

        channel_name = channels[channel_index - 1] if 0 <= channel_index - 1 < len(channels) else None

        for li in channel_ul.xpath("./li"):
            link = li.cssselect("a.title_link")
            href = link[0].get("href") if link else None
            abs_url = urljoin(BASE_URL, href) if href else None

            data_content = link[0].get("data-content") if link else None
            data_json = None
            if data_content:
                try:
                    data_json = json.loads(data_content)
                except json.JSONDecodeError:
                    data_json = None

            time_node = li.cssselect("div.program_time")
            time_text = " ".join(time_node[0].itertext()).strip() if time_node else None
            genre_class = None
            if time_node:
                class_tokens = (time_node[0].get("class") or "").split()
                genre_tokens = [t for t in class_tokens if t.startswith("gc-")]
                genre_class = genre_tokens[0] if genre_tokens else None

            title_node = li.cssselect("p.program_title")
            detail_node = li.cssselect("p.program_detail")

            style = li.get("style")
            top_px = None
            height_px = None
            if style:
                for kv in style.split(";"):
                    if kv.startswith("top:") and kv.endswith("px"):
                        top_px = int(kv.replace("top:", "").replace("px", "").strip())
                    if kv.startswith("height:") and kv.endswith("px"):
                        height_px = int(kv.replace("height:", "").replace("px", "").strip())

            event_id = None
            if href:
                path = urlparse(href).path.rstrip("/")
                if "/" in path:
                    event_id = path.rsplit("/", 1)[-1] or None

            rows.append(
                {
                    "source_type": req.source_type,
                    "broadcast_date": req.broadcast_date,
                    "ggm_group_id": req.ggm_group_id,
                    "channel_index": channel_index,
                    "channel_name": channel_name,
                    "event_id": event_id,
                    "event_url": abs_url,
                    "li_program_id": li.get("pid"),
                    "li_service_event_id": li.get("se-id"),
                    "li_start_at": li.get("s"),
                    "li_end_at": li.get("e"),
                    "slot_minute": time_text,
                    "title": " ".join(title_node[0].itertext()).strip() if title_node else None,
                    "detail": " ".join(detail_node[0].itertext()).strip() if detail_node else None,
                    "schedule_class": li.get("class"),
                    "genre_class": genre_class,
                    "style_top_px": top_px,
                    "style_height_px": height_px,
                    "metadata_title": data_json.get("title") if data_json else None,
                    "metadata_contents_id": data_json.get("contentsId") if data_json else None,
                    "metadata_program_id": data_json.get("programId") if data_json else None,
                    "metadata_program_date": data_json.get("programDate") if data_json else None,
                    "metadata_href": href,
                    "metadata_detail": " ".join(detail_node[0].itertext()).strip() if detail_node else None,
                }
            )
    return rows


async def init_db(db: aiosqlite.Connection) -> None:
    await db.execute("DROP TABLE IF EXISTS broadcast_events")
    await ensure_db_schema(db)
    await db.commit()


async def ensure_db_schema(db: aiosqlite.Connection) -> None:
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS broadcast_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT NOT NULL,
            broadcast_date TEXT NOT NULL,
            ggm_group_id INTEGER,
            channel_index INTEGER,
            channel_name TEXT,
            event_id TEXT,
            event_url TEXT,
            li_program_id TEXT,
            li_service_event_id TEXT,
            li_start_at TEXT,
            li_end_at TEXT,
            slot_minute TEXT,
            title TEXT,
            detail TEXT,
            schedule_class TEXT,
            genre_class TEXT,
            style_top_px INTEGER,
            style_height_px INTEGER,
            metadata_title TEXT,
            metadata_contents_id INTEGER,
            metadata_program_id TEXT,
            metadata_program_date TEXT,
            metadata_href TEXT,
            metadata_detail TEXT,
            detailed_description TEXT NOT NULL DEFAULT '',
            detail_fetched_at TEXT,
            inserted_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_broadcast_events_lookup ON broadcast_events(source_type, broadcast_date, ggm_group_id)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_broadcast_events_detail_fetch ON broadcast_events(detail_fetched_at, li_start_at)"
    )
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS fetch_status (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            last_broadcast_events_fetched_at TEXT,
            last_event_detail_fetched_at TEXT
        )
        """
    )
    await db.execute(
        """
        INSERT INTO fetch_status (id, last_broadcast_events_fetched_at, last_event_detail_fetched_at)
        VALUES (1, NULL, NULL)
        ON CONFLICT(id) DO NOTHING
        """
    )


async def set_last_broadcast_events_fetched_at(db: aiosqlite.Connection) -> None:
    await db.execute(
        "UPDATE fetch_status SET last_broadcast_events_fetched_at = datetime('now') WHERE id = 1"
    )


async def set_last_event_detail_fetched_at(db: aiosqlite.Connection) -> None:
    await db.execute(
        "UPDATE fetch_status SET last_event_detail_fetched_at = datetime('now') WHERE id = 1"
    )


async def store_rows(db: aiosqlite.Connection, req: SourceRequest, rows: list[dict]) -> None:
    await db.execute(
        "DELETE FROM broadcast_events WHERE source_type = ? AND broadcast_date = ? AND ((ggm_group_id IS NULL AND ? IS NULL) OR ggm_group_id = ?)",
        (req.source_type, req.broadcast_date, req.ggm_group_id, req.ggm_group_id),
    )

    if rows:
        await db.executemany(
            """
            INSERT INTO broadcast_events (
                source_type, broadcast_date, ggm_group_id,
                channel_index, channel_name,
                event_id, event_url,
                li_program_id, li_service_event_id,
                li_start_at, li_end_at,
                slot_minute, title, detail,
                schedule_class, genre_class,
                style_top_px, style_height_px,
                    metadata_title, metadata_contents_id,
                    metadata_program_id, metadata_program_date,
                    metadata_href, metadata_detail,
                    detailed_description
            ) VALUES (
                :source_type, :broadcast_date, :ggm_group_id,
                :channel_index, :channel_name,
                :event_id, :event_url,
                :li_program_id, :li_service_event_id,
                :li_start_at, :li_end_at,
                :slot_minute, :title, :detail,
                :schedule_class, :genre_class,
                :style_top_px, :style_height_px,
                    :metadata_title, :metadata_contents_id,
                    :metadata_program_id, :metadata_program_date,
                    :metadata_href, :metadata_detail,
                    ''
            )
            """,
            rows,
        )

    await db.commit()


async def collect(date: str, db_path: str, timeout: int) -> None:
    requests: list[SourceRequest] = [
        SourceRequest("terrestrial", f"{BASE_URL}/epg/td?broad_cast_date={date}&ggm_group_id={gid}", date, gid)
        for gid in TERRESTRIAL_GROUPS
    ] + [
        SourceRequest("bs", f"{BASE_URL}/epg/bs?broad_cast_date={date}", date),
        SourceRequest("cs", f"{BASE_URL}/epg/cs?broad_cast_date={date}", date),
    ]

    timeout_conf = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=timeout_conf) as session, aiosqlite.connect(db_path) as db:
        await init_db(db)

        html_results = await asyncio.gather(*(fetch_html(session, req) for req in requests))
        for req, source_html in zip(requests, html_results, strict=True):
            rows = parse_event_rows(req, source_html)
            await store_rows(db, req, rows)
            area = f" group={req.ggm_group_id}" if req.ggm_group_id is not None else ""
            print(f"stored {len(rows):4d} rows for {req.source_type}{area}")

        await set_last_broadcast_events_fetched_at(db)
        await db.commit()


async def print_stored_events(
    db_path: str,
    source_type: Optional[str],
    ggm_group_id: Optional[int],
    limit: int,
) -> None:
    query = """
        SELECT
            source_type,
            broadcast_date,
            ggm_group_id,
            channel_index,
            channel_name,
            metadata_title,
            li_start_at,
            li_end_at,
            event_url
        FROM broadcast_events
    """
    clauses: list[str] = []
    values: list[object] = []
    if source_type is not None:
        clauses.append("source_type = ?")
        values.append(source_type)
    if ggm_group_id is not None:
        clauses.append("ggm_group_id = ?")
        values.append(ggm_group_id)

    if clauses:
        query += " WHERE " + " AND ".join(clauses)

    query += " ORDER BY broadcast_date DESC, source_type, channel_index, li_start_at LIMIT ?"
    values.append(limit)

    async with aiosqlite.connect(db_path) as db:
        await ensure_db_schema(db)
        db.row_factory = aiosqlite.Row
        async with db.execute(query, values) as cursor:
            rows = await cursor.fetchall()

    if not rows:
        print("no rows found")
        return

    for row in rows:
        print(
            json.dumps(
                {
                    "source_type": row["source_type"],
                    "broadcast_date": row["broadcast_date"],
                    "ggm_group_id": row["ggm_group_id"],
                    "channel_index": row["channel_index"],
                    "channel_name": row["channel_name"],
                    "title": row["metadata_title"],
                    "start_at": row["li_start_at"],
                    "end_at": row["li_end_at"],
                    "event_url": row["event_url"],
                },
                ensure_ascii=False,
            )
        )


def parse_detailed_description(raw_html: str) -> str:
    tree = html.fromstring(raw_html)
    detail = tree.cssselect("#ggb_program_detail > p.letter_body")
    if not detail:
        return ""
    return " ".join(detail[0].itertext()).strip()


async def fetch_event_details(db_path: str, limit: int, timeout: int) -> None:
    async with aiosqlite.connect(db_path) as db:
        await ensure_db_schema(db)
        db.row_factory = aiosqlite.Row

        async with db.execute(
            "SELECT COUNT(*) AS c FROM broadcast_events WHERE detail_fetched_at IS NULL"
        ) as cursor:
            pending_count_row = await cursor.fetchone()
        pending_count = pending_count_row["c"] if pending_count_row else 0
        if pending_count == 0:
            print("all items already have detailed information retrieved")
            return

        async with db.execute(
            """
            SELECT id, event_url
            FROM broadcast_events
            WHERE event_url IS NOT NULL
            ORDER BY COALESCE(detail_fetched_at, '1970-01-01 00:00:00') ASC, li_start_at ASC
            LIMIT ?
            """,
            (limit,),
        ) as cursor:
            targets = await cursor.fetchall()

        if not targets:
            print("no rows found")
            return

        timeout_conf = aiohttp.ClientTimeout(total=timeout)
        fetched_count = 0
        async with aiohttp.ClientSession(timeout=timeout_conf) as session:
            for row in targets:
                async with session.get(row["event_url"]) as response:
                    response.raise_for_status()
                    raw_html = await response.text()

                detailed_description = parse_detailed_description(raw_html)
                await db.execute(
                    """
                    UPDATE broadcast_events
                    SET detailed_description = ?, detail_fetched_at = datetime('now')
                    WHERE id = ?
                    """,
                    (detailed_description, row["id"]),
                )
                print(f"fetched detailed description for id={row['id']}")
                fetched_count += 1

        if fetched_count > 0:
            await set_last_event_detail_fetched_at(db)

        await db.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bangumi checker CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser(
        "fetch-broadcast-events",
        help="Fetch broadcast events and store them in SQLite",
    )
    fetch_parser.add_argument("--date", help="Broadcast date (YYYYMMDD). Defaults to today")
    fetch_parser.add_argument("--db", default="broadcast_events.sqlite3", help="SQLite DB path")
    fetch_parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds")

    show_parser = subparsers.add_parser(
        "show-stored-events",
        help="Print stored events from SQLite",
    )
    show_parser.add_argument("--db", default="broadcast_events.sqlite3", help="SQLite DB path")
    show_parser.add_argument("--source-type", choices=["terrestrial", "bs", "cs"], help="Filter by source type")
    show_parser.add_argument("--ggm-group-id", type=int, help="Filter by terrestrial ggm_group_id")
    show_parser.add_argument("--limit", type=int, default=100, help="Max rows to print")

    fetch_detail_parser = subparsers.add_parser(
        "fetch-broadcast-event-details",
        help="Fetch detailed description for stored broadcast events",
    )
    fetch_detail_parser.add_argument("--db", default="broadcast_events.sqlite3", help="SQLite DB path")
    fetch_detail_parser.add_argument("--limit", type=int, default=50, help="Max rows to fetch")
    fetch_detail_parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "fetch-broadcast-events":
        date = args.date or datetime.date.today().strftime("%Y%m%d")
        if len(date) != 8 or not date.isdigit():
            raise SystemExit("--date must be YYYYMMDD")
        asyncio.run(collect(date, args.db, args.timeout))
        return

    if args.command == "show-stored-events":
        if args.limit <= 0:
            raise SystemExit("--limit must be greater than 0")
        asyncio.run(print_stored_events(args.db, args.source_type, args.ggm_group_id, args.limit))
        return

    if args.command == "fetch-broadcast-event-details":
        if args.limit <= 0:
            raise SystemExit("--limit must be greater than 0")
        asyncio.run(fetch_event_details(args.db, args.limit, args.timeout))
        return

    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
