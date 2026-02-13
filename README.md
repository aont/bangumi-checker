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
