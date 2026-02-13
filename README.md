# bangumi-checker

`main.py` provides a CLI to fetch program events from bangumi.org EPG pages, store them in SQLite, and fetch each event's detailed description.

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

Evaluate stored events with user-provided code and print only events where `evaluate_event(metadata)` returns `True`. Only events that have not previously matched (`user_function_returned_true = 0`) are checked:

```bash
python main.py evaluate-broadcast-events --db broadcast_events.sqlite3 --code-path ./my_filter.py
```

`my_filter.py` must define:

```python
def evaluate_event(metadata: dict) -> bool:
    return "アニメ" in (metadata.get("metadata_title") or "")
```

The evaluator loads user code directly from the provided file path with cache invalidation and updates per-event execution state columns in SQLite:

- `user_function_returned_true`
- `user_function_returned_false`
- `user_function_never_executed`


Optionally, `my_filter.py` can define `handle_matched_event(program)` to receive the full program row for each match (for notifications such as Slack integrations).
