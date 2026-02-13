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

Evaluate stored events with user-provided async code and print only events where `await evaluate_event(metadata)` returns `True`. By default, only events that have not previously matched (`user_function_returned_true = 0`) are checked:

```bash
python main.py evaluate-broadcast-events --db broadcast_events.sqlite3 --code-path ./my_filter.py
```

To force re-check all stored events regardless of previous `evaluate_event` results, add `--force`:

```bash
python main.py evaluate-broadcast-events --db broadcast_events.sqlite3 --code-path ./my_filter.py --force
```

Run periodic updates (every 3 hours by default), collecting broadcast event lists from today through one week ahead, then fetch event details and run the user-defined evaluator:

```bash
python main.py periodic-update --db broadcast_events.sqlite3 --code-path ./my_filter.py
```

Run a single update/evaluate cycle and exit:

```bash
python main.py periodic-update --db broadcast_events.sqlite3 --code-path ./my_filter.py --once
```

`my_filter.py` must define:

```python
async def evaluate_event(metadata: dict) -> bool:
    return "アニメ" in (metadata.get("metadata_title") or "")
```


A sample filter file is included as `my_filter.py`.
It matches programs whose title includes `🈟` or `🈡`, accumulates matched message text in `handle_matched_event`, and sends one combined Slack Incoming Webhook notification in `after_evaluate_events` when `SLACK_WEBHOOK_URL` is set.

The evaluator loads user code directly from the provided file path with cache invalidation and updates per-event execution state columns in SQLite:

- `user_function_returned_true`
- `user_function_returned_false`
- `user_function_never_executed`


Optionally, `my_filter.py` can define these async hooks:

- `before_evaluate_events()`: runs once before the first `evaluate_event` call.
- `handle_matched_event(program)`: receives the full program row for each match (for notifications such as Slack integrations).
- `after_evaluate_events()`: runs once after all `evaluate_event` calls have completed.
