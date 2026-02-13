#!/usr/bin/env python3
import argparse
import asyncio
import contextlib
import datetime
import importlib
import importlib.util
import inspect
import json
import logging
import pathlib
import random
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
LOG_LEVEL_CHOICES = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LOGGER = logging.getLogger(__name__)
SQLITE_BUSY_TIMEOUT_MS = 30_000


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@contextlib.asynccontextmanager
async def connect_db(db_path: str):
    db = await aiosqlite.connect(db_path, timeout=SQLITE_BUSY_TIMEOUT_MS / 1000)
    try:
        await db.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
        await db.execute("PRAGMA journal_mode = WAL")
        await db.execute("PRAGMA synchronous = NORMAL")
        yield db
    finally:
        await db.close()


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


async def sleep_request_interval() -> None:
    await asyncio.sleep(random.uniform(2, 7))


async def sleep_detail_request_interval() -> None:
    await asyncio.sleep(random.uniform(10, 20))


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
            needs_user_evaluation INTEGER NOT NULL DEFAULT 1,
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
    if "needs_user_evaluation" not in existing_columns:
        await db.execute(
            "ALTER TABLE broadcast_events ADD COLUMN needs_user_evaluation INTEGER NOT NULL DEFAULT 1"
        )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_broadcast_events_lookup ON broadcast_events(source_type, broadcast_date, ggm_group_id)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_broadcast_events_detail_fetch ON broadcast_events(detail_fetched_at, li_start_at)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_broadcast_events_eval_queue ON broadcast_events(needs_user_evaluation, broadcast_date)"
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


EVENT_COMPARE_FIELDS = [
    "channel_name",
    "event_id",
    "event_url",
    "li_end_at",
    "slot_minute",
    "title",
    "detail",
    "schedule_class",
    "genre_class",
    "style_top_px",
    "style_height_px",
    "metadata_title",
    "metadata_contents_id",
    "metadata_program_id",
    "metadata_program_date",
    "metadata_href",
    "metadata_detail",
]


def row_identity_key(row: dict) -> tuple:
    return (
        row.get("event_id") or "",
        row.get("li_service_event_id") or "",
        row.get("li_program_id") or "",
        row.get("li_start_at") or "",
        row.get("channel_index") or -1,
    )


def has_row_changed(new_row: dict, existing_row: aiosqlite.Row) -> bool:
    for key in EVENT_COMPARE_FIELDS:
        if new_row.get(key) != existing_row[key]:
            return True
    return False


async def store_rows(db: aiosqlite.Connection, req: SourceRequest, rows: list[dict]) -> None:
    async with db.execute(
        """
        SELECT *
        FROM broadcast_events
        WHERE source_type = ?
          AND broadcast_date = ?
          AND ((ggm_group_id IS NULL AND ? IS NULL) OR ggm_group_id = ?)
        """,
        (req.source_type, req.broadcast_date, req.ggm_group_id, req.ggm_group_id),
    ) as cursor:
        existing_rows = await cursor.fetchall()

    existing_by_key = {row_identity_key(dict(row)): row for row in existing_rows}
    incoming_keys = {row_identity_key(row) for row in rows}

    for key, existing_row in existing_by_key.items():
        if key in incoming_keys:
            continue
        await db.execute("DELETE FROM broadcast_events WHERE id = ?", (existing_row["id"],))

    for row in rows:
        key = row_identity_key(row)
        existing_row = existing_by_key.get(key)
        if existing_row is None:
            await db.execute(
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
                    detailed_description,
                    needs_user_evaluation
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
                    '',
                    1
                )
                """,
                row,
            )
            continue

        if has_row_changed(row, existing_row):
            await db.execute(
                """
                UPDATE broadcast_events
                SET channel_name = :channel_name,
                    event_id = :event_id,
                    event_url = :event_url,
                    li_program_id = :li_program_id,
                    li_service_event_id = :li_service_event_id,
                    li_start_at = :li_start_at,
                    li_end_at = :li_end_at,
                    slot_minute = :slot_minute,
                    title = :title,
                    detail = :detail,
                    schedule_class = :schedule_class,
                    genre_class = :genre_class,
                    style_top_px = :style_top_px,
                    style_height_px = :style_height_px,
                    metadata_title = :metadata_title,
                    metadata_contents_id = :metadata_contents_id,
                    metadata_program_id = :metadata_program_id,
                    metadata_program_date = :metadata_program_date,
                    metadata_href = :metadata_href,
                    metadata_detail = :metadata_detail,
                    detail_fetched_at = NULL,
                    detailed_description = '',
                    user_function_returned_true = 0,
                    user_function_returned_false = 0,
                    user_function_never_executed = 1,
                    needs_user_evaluation = 1
                WHERE id = :id
                """,
                {**row, "id": existing_row["id"]},
            )

    await db.commit()


async def collect(date: str, db_path: str, timeout: int, ggm_group_ids: list[int]) -> None:
    await collect_for_dates([date], db_path, timeout, ggm_group_ids)


async def collect_for_dates(dates: list[str], db_path: str, timeout: int, ggm_group_ids: list[int]) -> None:
    target_group_ids = ggm_group_ids or DEFAULT_GGM_GROUP_IDS
    requests: list[SourceRequest] = []
    for date in dates:
        requests.extend(
            [
                SourceRequest("terrestrial", f"{BASE_URL}/epg/td?broad_cast_date={date}&ggm_group_id={gid}", date, gid)
                for gid in target_group_ids
            ]
        )
        requests.extend(
            [
                SourceRequest("bs", f"{BASE_URL}/epg/bs?broad_cast_date={date}", date),
                SourceRequest("cs", f"{BASE_URL}/epg/cs?broad_cast_date={date}", date),
            ]
        )

    timeout_conf = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=timeout_conf) as session, connect_db(db_path) as db:
        await ensure_db_schema(db)
        db.row_factory = aiosqlite.Row

        for index, req in enumerate(requests):
            if index > 0:
                await sleep_request_interval()

            source_html = await fetch_html(session, req)
            rows = parse_event_rows(req, source_html)
            await store_rows(db, req, rows)
            area = f" group={req.ggm_group_id}" if req.ggm_group_id is not None else ""
            LOGGER.info("stored %4d rows for %s%s", len(rows), req.source_type, area)

        await set_last_broadcast_events_fetched_at(db)
        await db.commit()


def parse_detailed_description(raw_html: str) -> str:
    tree = html.fromstring(raw_html)
    detail = tree.cssselect("#ggb_program_detail > p.letter_body")
    if not detail:
        return ""
    return " ".join(detail[0].itertext()).strip()


async def fetch_event_details(db_path: str, timeout: int, limit: Optional[int] = None) -> int:
    async with connect_db(db_path) as db:
        await ensure_db_schema(db)
        db.row_factory = aiosqlite.Row

        async with db.execute(
            "SELECT COUNT(*) AS c FROM broadcast_events WHERE detail_fetched_at IS NULL"
        ) as cursor:
            pending_count_row = await cursor.fetchone()
        pending_count = pending_count_row["c"] if pending_count_row else 0
        if pending_count == 0:
            LOGGER.info("all items already have detailed information retrieved")
            return 0

        detail_select_sql = """
            SELECT id, event_url
            FROM broadcast_events
            WHERE event_url IS NOT NULL AND detail_fetched_at IS NULL
            ORDER BY COALESCE(detail_fetched_at, '1970-01-01 00:00:00') ASC, li_start_at ASC
        """
        if limit is not None:
            detail_select_sql += " LIMIT ?"
            async with db.execute(detail_select_sql, (limit,)) as cursor:
                targets = await cursor.fetchall()
        else:
            async with db.execute(detail_select_sql) as cursor:
                targets = await cursor.fetchall()

        if not targets:
            LOGGER.info("no rows found")
            return 0

        timeout_conf = aiohttp.ClientTimeout(total=timeout)
        fetched_count = 0
        async with aiohttp.ClientSession(timeout=timeout_conf) as session:
            for index, row in enumerate(targets):
                if index > 0:
                    await sleep_detail_request_interval()

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
                LOGGER.info("fetched detailed description for id=%s", row["id"])
                fetched_count += 1

        if fetched_count > 0:
            await set_last_event_detail_fetched_at(db)

        await db.commit()
        return fetched_count


def upcoming_dates(days_ahead: int = 7) -> list[str]:
    today = datetime.date.today()
    return [(today + datetime.timedelta(days=offset)).strftime("%Y%m%d") for offset in range(days_ahead + 1)]


async def periodic_update_and_evaluate(
    db_path: str,
    timeout: int,
    ggm_group_ids: list[int],
    code_path: str,
    interval_hours: int,
    run_once: bool,
) -> None:
    async def detail_fetch_loop() -> None:
        while True:
            fetched_count = await fetch_event_details(db_path, timeout)
            if run_once:
                return
            if fetched_count == 0:
                LOGGER.info("detail fetch worker idle; sleeping for 5 minutes")
                await asyncio.sleep(5 * 60)

    detail_worker = asyncio.create_task(detail_fetch_loop())
    try:
        while True:
            dates = upcoming_dates(7)
            LOGGER.info("starting update cycle for dates %s to %s", dates[0], dates[-1])
            for date in dates:
                await collect_for_dates([date], db_path, timeout, ggm_group_ids)
                await evaluate_broadcast_events(db_path, code_path, broadcast_dates=[date])
            if run_once:
                await detail_worker
                return
            LOGGER.info("cycle complete; sleeping for %s hours", interval_hours)
            await asyncio.sleep(interval_hours * 60 * 60)
    finally:
        if not detail_worker.done():
            detail_worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await detail_worker


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

    before_evaluate_events = getattr(module, "before_evaluate_events", None)
    if before_evaluate_events is not None and not callable(before_evaluate_events):
        raise SystemExit("before_evaluate_events must be callable when defined")
    if before_evaluate_events is not None and not inspect.iscoroutinefunction(before_evaluate_events):
        raise SystemExit("before_evaluate_events must be defined as async def when provided")

    after_evaluate_events = getattr(module, "after_evaluate_events", None)
    if after_evaluate_events is not None and not callable(after_evaluate_events):
        raise SystemExit("after_evaluate_events must be callable when defined")
    if after_evaluate_events is not None and not inspect.iscoroutinefunction(after_evaluate_events):
        raise SystemExit("after_evaluate_events must be defined as async def when provided")

    return evaluate_event, handle_matched_event, before_evaluate_events, after_evaluate_events


async def evaluate_broadcast_events(
    db_path: str,
    code_path: str,
    force: bool = False,
    broadcast_dates: Optional[list[str]] = None,
) -> None:
    user_code = pathlib.Path(code_path)
    if not user_code.exists() or not user_code.is_file():
        raise SystemExit(f"--code-path must point to an existing file: {code_path}")

    evaluate_event, handle_matched_event, before_evaluate_events, after_evaluate_events = load_user_functions(
        str(user_code.resolve())
    )

    async with connect_db(db_path) as db:
        await ensure_db_schema(db)
        db.row_factory = aiosqlite.Row

        where_conditions: list[str] = []
        params: list[object] = []
        if not force:
            where_conditions.append("needs_user_evaluation = 1")
        if broadcast_dates:
            placeholders = ",".join("?" for _ in broadcast_dates)
            where_conditions.append(f"broadcast_date IN ({placeholders})")
            params.extend(broadcast_dates)
        where_clause = f"WHERE {' AND '.join(where_conditions)}" if where_conditions else ""
        async with db.execute(
            f"""
            SELECT *
            FROM broadcast_events
            {where_clause}
            ORDER BY source_type, COALESCE(ggm_group_id, -1), channel_index, li_start_at
            """,
            params,
        ) as cursor:
            rows = await cursor.fetchall()

        matched_count = 0
        if before_evaluate_events is not None:
            await before_evaluate_events()

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
                    user_function_never_executed = 0,
                    needs_user_evaluation = 0
                WHERE id = ?
                """,
                (1 if result else 0, 0 if result else 1, row["id"]),
            )

            if result:
                matched_count += 1
                program = {k: row[k] for k in row.keys()}
                if handle_matched_event is not None:
                    await handle_matched_event(program)
                LOGGER.info(
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

        if after_evaluate_events is not None:
            await after_evaluate_events()

        await db.commit()
        summary_suffix = "(forced re-check enabled)" if force else "(excluding previously matched events)"
        LOGGER.info("matched %s / %s events checked %s", matched_count, len(rows), summary_suffix)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bangumi checker CLI")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVEL_CHOICES,
        help="Log level (default: INFO)",
    )
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

    periodic_parser = subparsers.add_parser(
        "periodic-update",
        aliases=["watch", "periodic"],
        help="Periodically refresh one week of events and run user checks while details are fetched asynchronously",
    )
    periodic_parser.add_argument("--db", default="broadcast_events.sqlite3", help="SQLite DB path")
    periodic_parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds")
    periodic_parser.add_argument(
        "--ggm-group-id",
        dest="ggm_group_ids",
        type=int,
        action="append",
        default=None,
        help="Terrestrial ggm_group_id (repeatable). Defaults to Tokyo only (42)",
    )
    periodic_parser.add_argument(
        "--code-path",
        required=True,
        help="Path to Python file defining async evaluate_event(metadata) -> bool",
    )
    periodic_parser.add_argument(
        "--interval-hours",
        type=int,
        default=3,
        help="Cycle interval in hours",
    )
    periodic_parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single update/evaluate cycle and exit",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
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
        asyncio.run(fetch_event_details(args.db, args.timeout, args.limit))
        return

    if args.command in {"evaluate-broadcast-events", "eval"}:
        asyncio.run(evaluate_broadcast_events(args.db, args.code_path, force=args.force))
        return

    if args.command in {"periodic-update", "watch", "periodic"}:
        group_ids = args.ggm_group_ids if args.ggm_group_ids is not None else DEFAULT_GGM_GROUP_IDS
        invalid_group_ids = [gid for gid in group_ids if gid not in TERRESTRIAL_GROUPS]
        if invalid_group_ids:
            supported = ", ".join(str(gid) for gid in TERRESTRIAL_GROUPS)
            invalid = ", ".join(str(gid) for gid in invalid_group_ids)
            raise SystemExit(f"unsupported --ggm-group-id: {invalid} (supported: {supported})")
        if args.interval_hours <= 0:
            raise SystemExit("--interval-hours must be greater than 0")
        asyncio.run(
            periodic_update_and_evaluate(
                db_path=args.db,
                timeout=args.timeout,
                ggm_group_ids=group_ids,
                code_path=args.code_path,
                interval_hours=args.interval_hours,
                run_once=args.once,
            )
        )
        return

    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
