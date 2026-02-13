# bangumi-checker

`fetch_broadcast_events.py` retrieves program events from bangumi.org EPG pages and stores them in SQLite.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
python fetch_broadcast_events.py --date 20260213 --db broadcast_events.sqlite3
```

The script fetches:

- Terrestrial (`ggm_group_id`: 37, 40, 42, 45)
- BS
- CS

For each page it parses `#contents > #program_area > ul > li` and stores extracted metadata.
When run again with the same date/source, previous rows are deleted and replaced.
