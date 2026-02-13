#!/usr/bin/env python3
import argparse
import asyncio
import datetime
import importlib
import importlib.util
import inspect
import json
import pathlib
import sys
import uuid
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
DEFAULT_GGM_GROUP_IDS = [42]


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
            user_function_returned_true INTEGER NOT NULL DEFAULT 0,
            user_function_returned_false INTEGER NOT NULL DEFAULT 0,
            user_function_never_executed INTEGER NOT NULL DEFAULT 1,
            detailed_description TEXT NOT NULL DEFAULT '',
            detail_fetched_at TEXT,
            inserted_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    async with db.execute("PRAGMA table_info(broadcast_events)") as cursor:
        existing_columns = {row[1] for row in await cursor.fetchall()}
    if "user_function_returned_true" not in existing_columns:
        await db.execute(
            "ALTER TABLE broadcast_events ADD COLUMN user_function_returned_true INTEGER NOT NULL DEFAULT 0"
        )
    if "user_function_returned_false" not in existing_columns:
        await db.execute(
            "ALTER TABLE broadcast_events ADD COLUMN user_function_returned_false INTEGER NOT NULL DEFAULT 0"
        )
    if "user_function_never_executed" not in existing_columns:
        await db.execute(
            "ALTER TABLE broadcast_events ADD COLUMN user_function_never_executed INTEGER NOT NULL DEFAULT 1"
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


async def collect(date: str, db_path: str, timeout: int, ggm_group_ids: list[int]) -> None:
    target_group_ids = ggm_group_ids or DEFAULT_GGM_GROUP_IDS
    requests: list[SourceRequest] = [
        SourceRequest("terrestrial", f"{BASE_URL}/epg/td?broad_cast_date={date}&ggm_group_id={gid}", date, gid)
        for gid in target_group_ids
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


def load_user_functions(code_path: str):
    importlib.invalidate_caches()
    module_name = f"user_filter_{uuid.uuid4().hex}"
    module_spec = importlib.util.spec_from_file_location(module_name, code_path)
    if module_spec is None or module_spec.loader is None:
        raise SystemExit(f"failed to load user code from: {code_path}")

    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    original_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module_spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
        sys.dont_write_bytecode = original_dont_write_bytecode

    evaluate_event = getattr(module, "evaluate_event", None)
    if not callable(evaluate_event):
        raise SystemExit("user code must define callable: async evaluate_event(metadata) -> bool")
    if not inspect.iscoroutinefunction(evaluate_event):
        raise SystemExit("evaluate_event must be defined as async def")

    handle_matched_event = getattr(module, "handle_matched_event", None)
    if handle_matched_event is not None and not callable(handle_matched_event):
        raise SystemExit("handle_matched_event must be callable when defined")
    if handle_matched_event is not None and not inspect.iscoroutinefunction(handle_matched_event):
        raise SystemExit("handle_matched_event must be defined as async def when provided")

    return evaluate_event, handle_matched_event


async def evaluate_broadcast_events(db_path: str, code_path: str, force: bool = False) -> None:
    user_code = pathlib.Path(code_path)
    if not user_code.exists() or not user_code.is_file():
        raise SystemExit(f"--code-path must point to an existing file: {code_path}")

    evaluate_event, handle_matched_event = load_user_functions(str(user_code.resolve()))

    async with aiosqlite.connect(db_path) as db:
        await ensure_db_schema(db)
        db.row_factory = aiosqlite.Row

        where_clause = "" if force else "WHERE user_function_returned_true = 0"
        async with db.execute(
            f"""
            SELECT *
            FROM broadcast_events
            {where_clause}
            ORDER BY source_type, COALESCE(ggm_group_id, -1), channel_index, li_start_at
            """
        ) as cursor:
            rows = await cursor.fetchall()

        matched_count = 0
        for row in rows:
            metadata = {k: row[k] for k in row.keys() if k.startswith("metadata_")}
            result = await evaluate_event(metadata)
            if not isinstance(result, bool):
                raise SystemExit(f"evaluate_event must return bool (event id={row['id']})")

            await db.execute(
                """
                UPDATE broadcast_events
                SET user_function_returned_true = ?,
                    user_function_returned_false = ?,
                    user_function_never_executed = 0
                WHERE id = ?
                """,
                (1 if result else 0, 0 if result else 1, row["id"]),
            )

            if result:
                matched_count += 1
                program = {k: row[k] for k in row.keys()}
                if handle_matched_event is not None:
                    await handle_matched_event(program)
                print(
                    json.dumps(
                        {
                            "id": row["id"],
                            "source_type": row["source_type"],
                            "broadcast_date": row["broadcast_date"],
                            "channel_name": row["channel_name"],
                            "slot_minute": row["slot_minute"],
                            "title": row["title"],
                            "event_url": row["event_url"],
                            "metadata": metadata,
                        },
                        ensure_ascii=False,
                    )
                )

        await db.commit()
        summary_suffix = "(forced re-check enabled)" if force else "(excluding previously matched events)"
        print(f"matched {matched_count} / {len(rows)} events checked {summary_suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bangumi checker CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser(
        "fetch-broadcast-events",
        aliases=["fetch"],
        help="Fetch broadcast events and store them in SQLite",
    )
    fetch_parser.add_argument("--date", help="Broadcast date (YYYYMMDD). Defaults to today")
    fetch_parser.add_argument("--db", default="broadcast_events.sqlite3", help="SQLite DB path")
    fetch_parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds")
    fetch_parser.add_argument(
        "--ggm-group-id",
        dest="ggm_group_ids",
        type=int,
        action="append",
        default=None,
        help="Terrestrial ggm_group_id (repeatable). Defaults to Tokyo only (42)",
    )

    fetch_detail_parser = subparsers.add_parser(
        "fetch-broadcast-event-details",
        aliases=["fetch-broadcast-events-details", "detail"],
        help="Fetch detailed description for stored broadcast events",
    )
    fetch_detail_parser.add_argument("--db", default="broadcast_events.sqlite3", help="SQLite DB path")
    fetch_detail_parser.add_argument("--limit", type=int, default=50, help="Max rows to fetch")
    fetch_detail_parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds")

    evaluate_parser = subparsers.add_parser(
        "evaluate-broadcast-events",
        aliases=["eval"],
        help="Evaluate broadcast events with user code and print rows returning True",
    )
    evaluate_parser.add_argument("--db", default="broadcast_events.sqlite3", help="SQLite DB path")
    evaluate_parser.add_argument(
        "--code-path",
        required=True,
        help="Path to Python file defining async evaluate_event(metadata) -> bool",
    )
    evaluate_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-check all events regardless of previous evaluate_event results",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command in {"fetch-broadcast-events", "fetch"}:
        date = args.date or datetime.date.today().strftime("%Y%m%d")
        if len(date) != 8 or not date.isdigit():
            raise SystemExit("--date must be YYYYMMDD")
        group_ids = args.ggm_group_ids if args.ggm_group_ids is not None else DEFAULT_GGM_GROUP_IDS
        invalid_group_ids = [gid for gid in group_ids if gid not in TERRESTRIAL_GROUPS]
        if invalid_group_ids:
            supported = ", ".join(str(gid) for gid in TERRESTRIAL_GROUPS)
            invalid = ", ".join(str(gid) for gid in invalid_group_ids)
            raise SystemExit(f"unsupported --ggm-group-id: {invalid} (supported: {supported})")
        asyncio.run(collect(date, args.db, args.timeout, group_ids))
        return

    if args.command in {
        "fetch-broadcast-event-details",
        "fetch-broadcast-events-details",
        "detail",
    }:
        if args.limit <= 0:
            raise SystemExit("--limit must be greater than 0")
        asyncio.run(fetch_event_details(args.db, args.limit, args.timeout))
        return

    if args.command in {"evaluate-broadcast-events", "eval"}:
        asyncio.run(evaluate_broadcast_events(args.db, args.code_path, force=args.force))
        return

    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
