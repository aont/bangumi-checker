# Frontend / Backend Communication Notes

## Backend Address Selection

- The frontend uses the Base URL stored in `localStorage` under the key `bangumiChecker.backendBaseUrl`.
- If no Base URL is configured, it uses relative paths (`/api/...`), so requests go to a backend on the same origin as the frontend.
- The default `serve-watch` bind address is `127.0.0.1:8080`.

## APIs Called by the Frontend

- `GET /api/status`
- `GET /api/config`
- `PATCH /api/config`
- `GET /api/script`
- `PUT /api/script`
- `POST /api/script/validate`
- `POST /api/actions/run-once`
- `POST /api/actions/fetch-details`
- `POST /api/actions/evaluate`
- `GET /api/events?limit=30&offset=0`
- `GET /api/matches?limit=30&offset=0`

## Main Request JSON Payloads

- `PATCH /api/config`
  - `{"timeout": number, "interval_hours": number, "ggm_group_ids": number[], "code_path": string, "enabled": boolean}`
- `PUT /api/script`
  - `{"content": string}`
- `POST /api/script/validate`
  - `{"content": string}`
- `POST /api/actions/fetch-details`
  - `{"limit": 1}`
- `POST /api/actions/evaluate`
  - `{"force": false}`
- `POST /api/actions/run-once`
  - `{}`

## Representative Response Shapes

- `GET /api/status`
  - `phase`, `last_cycle_started_at`, `last_cycle_finished_at`, `last_error`
  - `config` (timeout / ggm_group_ids / code_path / interval_hours / enabled)
  - `db` (last_broadcast_events_fetched_at / last_event_detail_fetched_at / pending_evaluation_count / pending_detail_count)
- `GET /api/config`
  - timeout / ggm_group_ids / code_path / interval_hours / enabled
- `GET /api/script`
  - code_path / content
- `GET /api/events`, `GET /api/matches`
  - `{"items": [...], "limit": number, "offset": number}`

## CORS

- The backend returns permissive CORS headers such as `Access-Control-Allow-Origin: *`.
- It also responds to `OPTIONS` preflight requests with `204`.
