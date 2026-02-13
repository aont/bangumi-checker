# bangumi-checker

`main.py` provides a CLI to fetch program events from bangumi.org EPG pages, store them in SQLite, fetch each event's detailed description, and run periodic update/evaluation cycles.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

Fetch and store events (`--date` omitted then today):

```bash
python main.py fetch-broadcast-events --db broadcast_events.sqlite3
python main.py fetch-broadcast-events --date 20260213 --db broadcast_events.sqlite3
python main.py --log-level DEBUG fetch-broadcast-events --date 20260213 --db broadcast_events.sqlite3
```

The fetch command retrieves:

- Terrestrial (`ggm_group_id`: default is Tokyo `42`; can be overridden with `--ggm-group-id`)
- BS
- CS

Example with multiple terrestrial groups:

```bash
python main.py fetch-broadcast-events --date 20260213 --ggm-group-id 42 --ggm-group-id 45 --db broadcast_events.sqlite3
```

For each page it parses `#contents > #program_area > ul > li` and stores extracted metadata.
When run again with the same date/source, previous rows are deleted and replaced.

Fetch detailed descriptions for stored events:

```bash
python main.py fetch-broadcast-event-details --db broadcast_events.sqlite3 --limit 50
```

Detailed information is fetched one event at a time with a random 10-20 second interval between requests.

Evaluate stored events with user-provided async code and print only events where `await evaluate_event(metadata)` returns `True`. By default, only events flagged as needing evaluation (newly inserted or updated rows) are checked:

```bash
python main.py evaluate-broadcast-events --db broadcast_events.sqlite3 --code-path ./example/title_marker_filter.py
```

To force re-check all stored events regardless of previous `evaluate_event` results, add `--force`:

```bash
python main.py evaluate-broadcast-events --db broadcast_events.sqlite3 --code-path ./example/title_marker_filter.py --force
```

Run periodic updates (every 3 hours by default), collecting broadcast event lists from today through one week ahead. For each target day, the user-defined evaluator (including before/after hooks) runs right after that day's list retrieval completes. Event detail retrieval runs in a separate asynchronous worker and progresses independently for all pending programs:

```bash
python main.py periodic-update --db broadcast_events.sqlite3 --code-path ./example/title_marker_filter.py
```

Run a single update/evaluate cycle and exit:

```bash
python main.py periodic-update --db broadcast_events.sqlite3 --code-path ./example/title_marker_filter.py --once
```

`example/title_marker_filter.py` must define:

```python
async def evaluate_event(metadata: dict) -> bool:
    return "アニメ" in (metadata.get("metadata_title") or "")
```


Sample filters are included in the `example/` directory: `example/title_marker_filter.py` and `example/mentaiko_detail_filter.py`.
The latter matches programs whose detailed text includes `明太子` and sends Slack notifications in the same way.

It matches programs whose title includes `🈟` or `🈡`, accumulates matched message text in `handle_matched_event`, and sends one combined Slack Incoming Webhook notification in `after_evaluate_events` when `SLACK_WEBHOOK_URL` is set.

The evaluator loads user code directly from the provided file path with cache invalidation and updates per-event execution state columns in SQLite:

- `user_function_returned_true`
- `user_function_returned_false`
- `user_function_never_executed`


Optionally, `example/title_marker_filter.py` can define these async hooks:

- `before_evaluate_events()`: runs once before the first `evaluate_event` call.
- `handle_matched_event(program)`: receives the full program row for each match (for notifications such as Slack integrations).
- `after_evaluate_events()`: runs once after all `evaluate_event` calls have completed.


All logs include timestamps, and the global log level can be controlled via `--log-level` (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).

## User-defined function I/O specification

### `evaluate_event(metadata: dict) -> bool`

- **Input**: `metadata` is a dictionary that includes all columns from the `broadcast_events` row (`SELECT *` 相当)。
- **Output**: must return `True` or `False`.

`metadata` includes every `broadcast_events` column. Frequently used keys include:

| Key | Type | Description |
| --- | --- | --- |
| `metadata_title` | `str \| None` | Program title parsed from `data-content.title`. |
| `contents_id` | `int \| None` | Contents ID parsed from `data-content.contentsId`. |
| `metadata_program_id` | `str \| None` | Program ID parsed from `data-content.programId`. |
| `program_date` | `str \| None` | Program date parsed from `data-content.programDate`. |
| `href` | `str \| None` | Relative link from `a.title_link[href]` (example: `/tv_events/...`). |
| `metadata_detail` | `str \| None` | Summary/detail text extracted from `.detail` block on the list page. |
| `event_id` | `str \| None` | Event token parsed from the event URL. |
| `channel_name` | `str \| None` | Channel display name. |
| `detailed_description` | `str` | Detail page body text (empty string until fetched). |

### Optional hooks

#### `handle_matched_event(program: dict) -> None`

- **Input**: full SQLite row as a dictionary (`SELECT * FROM broadcast_events`).
- Useful when you need non-`metadata_*` fields such as `channel_name`, `li_start_at`, `title`, `event_url`, or `detailed_description`.

#### `before_evaluate_events() -> None` / `after_evaluate_events() -> None`

- No arguments, no return value.
- Run once before/after the evaluation loop.
