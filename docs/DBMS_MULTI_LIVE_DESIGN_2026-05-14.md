# DBMS Persistence and Multi-Live Design — 2026-05-14

Status: design note only; no implementation changes yet.

## Intent

Plan one combined refactor for two overlapping upgrades:

1. Replace the current filesystem persistence layer with DBMS-backed persistence.
2. Support multiple live trading instances/accounts without state, journal, dashboard, or deployment collisions.

The key design choice is to introduce account/run identity and persistence abstractions before changing storage internals, so DBMS migration and multi-live support do not become two separate rewrites.

## Current project context reviewed

### Repositories

Primary system:

- `rocky-d/cta-forge` at `/home/node/projects/quant/cta-forge`
  - Python 3.12 `uv` workspace.
  - Services: data-service, alpha-service, strategy-service, executor, report-service.
  - Shared libs: core, exchange.
  - Current storage stack: parquet cache, JSON state file, JSONL journals.

Observability/dashboard repos:

- `quant-dashboard-collector`
  - Go sidecar reads live JSONL journals and publishes sanitized latest/history payloads.
- `quant-dashboard-realtime`
  - Go sidecar reads equity journal and public Hyperliquid mids, then publishes realtime estimate overlay.
- `quant-dashboard-worker`
  - Cloudflare Worker stores latest/history in KV and realtime overlay in a Durable Object.
  - Current public strategy allowlist is `cta-forge` only.
- `quant-dashboard-frontend`
  - Reads fixed public endpoints for `cta-forge`: latest, history, realtime.

Supporting/research repos:

- `alt-coinmarketcap`, `alt-glassnode`, `alt-defillama`, `bnufsa`.
- These are relevant to future data integration, but not directly in scope for the first DBMS/multi-live refactor.

### GitHub Project context

Project: `Quantitative Research and Development in Crypto Market` (`rocky-d` Project #2).

Relevant backlog items already exist:

- `[Story] Replace filesystem journals and pipeline with PostgreSQL-backed persistence`
- `[Research] Evaluate multi-live account/wallet architecture options`
- `[Story] Implement and validate multi-live runtime and data isolation`
- `[Ops] Design capital rollout and activate additional live context when safe`

Relevant in-progress epics:

- Controlled mainnet pilot and live-risk operations.
- Public dashboard and observability.
- Delivery governance and Agile evidence.

The board direction matches this design note: phase-2 hardening first, then persistence/multi-live as small, reviewable increments with live-risk gates.

## Current persistence map

### Market data cache

Current implementation:

- `services/data-service/src/data_service/store.py`
- `ParquetStore` layout: `{DATA_DIR}/{SYMBOL}/{interval}.parquet`
- Used by:
  - data-service REST routes.
  - executor live data refresh.
  - v16a target construction.
  - research/backtest scripts.

Current behavior:

- Reads/writes polars DataFrames.
- Merges by `open_time` and keeps latest row.
- No DB identity, no source table, no audit metadata.

### Live engine state

Current implementation:

- `services/executor/src/executor/state.py`
- Default file: `engine-state.json`
- Stores:
  - bar count
  - initial/peak equity
  - DD breaker flag
  - last signals
  - recent returns
  - last tick equity
  - internal position state

Current behavior:

- Atomic file write via temp file + move.
- One state file per live runtime only if `STATE_FILE` is manually separated.
- State is reconciled with exchange positions on startup.

### Live journals

Current implementation:

- `services/executor/src/executor/journal.py`
- Files inside `JOURNAL_DIR`:
  - `equity.jsonl`
  - `trades.jsonl`
  - `signals.jsonl`
  - `targets.jsonl`

Used by:

- Live reports in executor API.
- Report-service live journal chart.
- Dashboard collector sidecar.
- Dashboard realtime sidecar.
- Forward-shadow diagnostics.

Current issue:

- Journal identity is implicit in path names such as `journal/mainnet-pilot`.
- Multi-live would require manually separated paths and sidecars.
- The files are append-only but not transactional across tick/target/trade/state writes.

### Runtime configuration

Current implementation:

- `executor.run_live` reads environment variables directly.
- Key identifiers and paths:
  - `HL_PRIVATE_KEY`, `HL_ACCOUNT_ADDRESS`
  - `HL_NETWORK`
  - `STRATEGY_PROFILE`
  - `STATE_FILE`, `JOURNAL_DIR`, `DATA_DIR`
  - risk caps and live symbols.

Current issue:

- Account identity is tied to wallet address and env file, not a first-class `account_id` / `instance_id`.
- Multiple live runtimes would be possible at the process level, but not cleanly represented in state, journals, dashboard, or deployment outputs.

## Current live pipeline map

Simplified flow:

1. `run_live.py`
   - Parse env.
   - Validate profile/network/risk gates.
   - Build Hyperliquid adapter.
   - Build `LiveEngine`.
2. `LiveEngine.start()`
   - Preflight.
   - Load state file.
   - Reconcile exchange positions.
   - Restore equity peak from journal.
3. `LiveEngine._tick()`
   - Refresh parquet bars for target strategy.
   - Read account state from exchange.
   - Update tick return and target/execution state.
   - Record target diagnostics.
   - Place target orders if allowed.
   - Record trades.
   - Record equity and signals.
   - Save state.
   - Notify.
4. Dashboard sidecars independently read JSONL files and publish public snapshots/history/realtime overlay.

Key point: tick execution, state persistence, journal writes, and dashboard ingestion are coupled by file paths rather than a stable domain model.

## External research notes

PostgreSQL is the recommended first DBMS target, not because it is trendy, but because it covers relational identity, transactional writes, JSONB diagnostics, time-series partitioning, uniqueness/upsert, and simple backup/restore in one operational surface.

Relevant official references:

- PostgreSQL declarative table partitioning supports range/list/hash partitioning and partition pruning for large tables: https://www.postgresql.org/docs/current/ddl-partitioning.html
- PostgreSQL JSON/JSONB supports structured semi-flexible diagnostic payloads; JSONB can be indexed with GIN for key/value search when needed: https://www.postgresql.org/docs/current/datatype-json.html and https://www.postgresql.org/docs/current/gin.html
- PostgreSQL `INSERT ... ON CONFLICT` supports idempotent upsert patterns when backed by unique constraints/indexes: https://www.postgresql.org/docs/current/sql-insert.html
- PostgreSQL advisory locks can represent application-defined locks such as `live_instance` ownership: https://www.postgresql.org/docs/current/functions-admin.html
- `pg_dump`/`pg_restore` provide a baseline logical backup/restore path: https://www.postgresql.org/docs/current/app-pgdump.html
- TimescaleDB hypertables are worth considering later for larger time-series workloads, but the first step should avoid mandatory extensions unless native PostgreSQL becomes insufficient: https://docs.timescale.com/use-timescale/latest/hypertables/

Design conclusion: start with native PostgreSQL + SQL migrations. Keep TimescaleDB as an optional later optimization, not an initial dependency.

## Design principles for this refactor

- Keep live-risk changes explicit and fail-closed.
- Preserve current live behavior before adding new behavior.
- Add identity and interfaces first; switch storage behind them second.
- Prefer small composition boundaries over a large ORM-driven rewrite.
- Do not make DBMS a hard requirement for research scripts on day one.
- Keep the current JSONL/parquet paths as migration/export compatibility until DB-backed behavior is validated.
- Avoid public dashboard leakage: DB may contain private account data; public payloads remain sanitized contracts.

## Proposed domain model

### Core identity

Introduce stable identifiers:

- `strategy_id`
  - Example: `cta-forge`.
  - Public/dashboard grouping.
- `profile_id`
  - Example: `v16a-mainnet-pilot`.
  - Strategy profile/config lineage.
- `account_id`
  - Internal stable label, not the full wallet address.
  - Example: `hl-mainnet-pilot-01`.
- `exchange_account_ref`
  - Encrypted or redacted reference to network/address.
  - Full address should remain private; public payloads never expose it.
- `live_instance_id`
  - One runtime process/account/profile/network tuple.
  - Example: `cta-forge-mainnet-pilot-01`.
- `run_id`
  - One process lifecycle / deployment run.
  - Created at engine startup.
- `tick_id`
  - One scheduled live tick.
  - Unique on `(live_instance_id, bar)` or `(live_instance_id, target_ts)` depending mode.

### Entity sketch

Reference tables:

- `strategies(id, slug, name, created_at)`
- `strategy_profiles(id, strategy_id, slug, version, config_json, created_at)`
- `exchange_accounts(id, exchange, network, account_label, address_hash, address_prefix, status, created_at)`
- `live_instances(id, strategy_id, profile_id, account_id, slug, mode, status, risk_config_json, public_enabled, created_at)`
- `live_runs(id, live_instance_id, git_sha, image_ref, started_at, stopped_at, runtime_config_json, status)`

Operational state:

- `engine_checkpoints(live_instance_id, run_id, bar_count, payload_json, saved_at)`
  - One latest checkpoint per instance.
  - Replaces `engine-state.json`.
- `live_ticks(id, live_instance_id, run_id, bar, ts, account_equity, peak_equity, dd_pct, n_positions, status, summary_json)`
  - Replaces `equity.jsonl` as primary equity timeline.
- `live_positions(tick_id, live_instance_id, symbol, side, qty, entry_price, best_price, raw_json)`
  - Snapshot rows derived from tick/account state.
- `live_targets(id, live_instance_id, run_id, bar, ts, profile, target_ts, staleness_seconds, target_gross, normalized_gross, ignored_gross, execution_coverage, weights_json, ignored_weights_json, orders_json)`
  - Replaces `targets.jsonl`.
- `live_trades(id, live_instance_id, run_id, bar, ts, kind, symbol, side, qty, price, reason, pnl, pnl_pct, held_bars, exchange_order_id, raw_json)`
  - Replaces `trades.jsonl`.
- `live_signals(id, live_instance_id, run_id, bar, ts, signals_json)`
  - Replaces `signals.jsonl`.

Market data:

- `market_bars(source, symbol, interval, open_time, close_time, open, high, low, close, volume, raw_json, ingested_at)`
  - Unique key: `(source, symbol, interval, open_time)`.
  - Native PostgreSQL range partition by `open_time` can be introduced after the basic table exists.
  - Keep parquet import/export until backtests are moved.

Public dashboard materialization:

- `public_snapshots(strategy_slug, live_instance_slug, schema_version, generated_at, payload_json)`
- `public_history(strategy_slug, live_instance_slug, schema_version, generated_at, payload_json)`
- `public_realtime_overlays(strategy_slug, live_instance_slug, schema_version, generated_at, payload_json)`

These can stay in the dashboard Worker for public delivery, but the source sidecar should read DB rows instead of JSONL once DB ingestion is stable.

## Storage abstraction proposal

Introduce small ports instead of spreading DB calls through strategy code:

- `MarketDataStore`
  - `read(symbol, interval, start=None, end=None) -> pl.DataFrame`
  - `write(symbol, interval, df) -> int`
  - `latest_timestamp(symbol, interval)`
  - `symbols()`
  - Implementations: `ParquetMarketDataStore`, later `PostgresMarketDataStore`.

- `LiveStateStore`
  - `load_checkpoint(instance_id) -> LiveState | None`
  - `save_checkpoint(instance_id, run_id, state)`
  - Implementations: `JsonFileLiveStateStore`, later `PostgresLiveStateStore`.

- `LiveJournalStore`
  - `record_tick(instance_id, run_id, ...)`
  - `record_target(instance_id, run_id, ...)`
  - `record_trade(instance_id, run_id, ...)`
  - `record_signals(instance_id, run_id, ...)`
  - `load_equity(instance_id, ...)`
  - `load_targets(instance_id, ...)`
  - `load_trades(instance_id, ...)`
  - Implementations: `JsonlLiveJournalStore`, later `PostgresLiveJournalStore`.

- `LiveInstanceRegistry`
  - Resolves env/config into `live_instance_id`, account metadata, public/dashboard slug, and risk config.
  - Owns startup registration and run record creation.

This keeps the current strategy/backtest logic mostly unchanged and isolates migration risk.

## Multi-live runtime model

### Recommended first model: one process/container per live instance

Do not start with one orchestrator process managing multiple accounts. The safer first implementation is multiple isolated live containers, each with:

- Its own `LIVE_INSTANCE_ID`.
- Its own account secret env.
- Its own risk config.
- Its own advisory lock in DB.
- Its own notification routing label.
- Its own public/private dashboard policy.

Why:

- Preserves current failure isolation.
- Avoids one event loop failure affecting all accounts.
- Easier emergency stop/rollback per account.
- Better fit for existing Docker/systemd/deploy model.

### DB-level runtime guard

At startup, each live runtime should acquire an ownership lock:

- Use a PostgreSQL advisory lock keyed by `live_instance_id`, or a lease row with TTL.
- If the lock cannot be acquired, the runtime exits fail-closed.
- This prevents accidental duplicate containers trading the same account/profile.

### Config shape

Move from only path/env identity to explicit instance identity:

```text
LIVE_INSTANCE_ID=cta-forge-mainnet-pilot-01
STRATEGY_ID=cta-forge
ACCOUNT_ID=hl-mainnet-pilot-01
PUBLIC_INSTANCE_SLUG=mainnet-pilot
DATABASE_URL=postgresql://...
PERSISTENCE_BACKEND=jsonl|postgres|dual
MARKET_DATA_BACKEND=parquet|postgres|dual
```

Existing risk env vars can remain initially. Later they can be moved into a structured instance config file/table, but not in the first DB migration.

### Dashboard route and pipeline shape

Dashboard multi-live switching is a downstream feature, but the upstream DB/API design must reserve the identity model now. The dashboard should not need to infer account identity from journal paths or deployment names.

Short-term compatibility:

- Keep current public route as the default instance alias:
  - `/api/v1/live/cta-forge/public/latest`
  - `/api/v1/live/cta-forge/public/history`
  - `/api/v1/live/cta-forge/public/realtime`
- The default route should resolve to one configured `public_instance_slug`, initially the existing mainnet pilot.
- Existing frontend behavior should stay unchanged until the instance switcher is deliberately enabled.

Multi-live extension to design now, implement later:

- Add instance discovery endpoint:
  - `/api/v1/live/cta-forge/public/instances`
- Add instance-aware read endpoints:
  - `/api/v1/live/cta-forge/instances/{public_instance_slug}/public/latest`
  - `/api/v1/live/cta-forge/instances/{public_instance_slug}/public/history`
  - `/api/v1/live/cta-forge/instances/{public_instance_slug}/public/realtime`
- Keep strategy-level default endpoints as aliases, not as the only canonical storage key.

Suggested public instance metadata payload:

```json
{
  "schema_version": "dashboard.public_instances.v1",
  "generated_at": "2026-05-14T00:00:00Z",
  "strategy": {"slug": "cta-forge"},
  "default_instance_slug": "mainnet-pilot",
  "instances": [
    {
      "slug": "mainnet-pilot",
      "display_name": "Mainnet Pilot",
      "status": "live",
      "visibility": "public",
      "started_at": "2026-05-04T10:00:00Z",
      "latest_tick_at": "2026-05-14T06:03:00Z"
    }
  ]
}
```

Public metadata constraints:

- Use human-safe labels and public slugs only.
- Do not expose internal `account_id`, wallet address, exchange order ids, private run ids, raw positions, balances, or host paths.
- Instance status can be coarse: `live`, `stale`, `paused`, `retired`, `hidden`.

Dashboard pipeline stages:

1. Collector/realtime sidecars read DB or DB-backed executor API by `public_instance_slug`.
2. Worker validates and stores payloads under keys including both strategy and instance slug.
3. Default strategy-only keys are aliases/copies of the configured default instance.
4. Frontend initially keeps one default view; later it fetches `/instances` and renders a switcher.
5. Switching instance changes latest/history/realtime endpoint URLs but not chart component contracts.

This makes multi-live dashboard support a downstream UI/product feature while ensuring the persistence and API layers are already instance-aware.

## Migration strategy

### Phase 0 — evidence and compatibility tests

- Freeze current JSONL/parquet schema expectations with fixtures.
- Add tests around:
  - journal load/record behavior;
  - state load/save behavior;
  - dashboard collector payloads;
  - target diagnostics semantics;
  - report-service live journal plotting.

No behavior change.

### Phase 1 — identity plumbing without DB cutover

- Add `LIVE_INSTANCE_ID` and `RUN_ID` to runtime config.
- Include instance/run identifiers in new journal records while preserving old fields.
- Keep JSONL files as source of truth.
- Dashboard collector should ignore unknown fields.
- Add path defaults derived from instance id only when explicit paths are absent.

Goal: multi-live can be simulated with separate paths and no DB yet.

### Phase 2 — storage interfaces and dual-write

- Introduce `LiveJournalStore` and `LiveStateStore` ports.
- Keep JSONL/file implementation as default.
- Add PostgreSQL implementation behind `PERSISTENCE_BACKEND=postgres|dual`.
- In `dual` mode:
  - Write JSONL + DB.
  - Read state/report from JSONL initially.
  - Compare DB-derived summaries against JSONL-derived summaries in tests/diagnostics.

Goal: validate DB writes without changing live source of truth.

### Phase 3 — DB read path for reports/dashboard

- Add executor live report routes backed by DB for a selected instance.
- Update dashboard collector to read DB rows or a DB-derived HTTP endpoint instead of JSONL paths.
- Keep JSONL export as operational fallback.

Goal: dashboard no longer depends on host file paths.

### Phase 4 — DB checkpoint source of truth

- Switch live state restore to DB checkpoint for selected instance.
- Keep periodic JSON checkpoint/export for rollback during the transition.
- Add migration command: JSONL/state import into DB for historical continuity.

Goal: DB becomes primary persistence for live runtime.

### Phase 5 — market bars migration

- Add `MarketDataStore` abstraction.
- Import parquet cache into PostgreSQL `market_bars`.
- Add PostgreSQL-backed data-service mode.
- Keep parquet read/export until research scripts are migrated or a local-cache policy is chosen.

Market bars are intentionally later because live journaling/state is the bigger multi-live blocker.

## Implementation impact map

Primary cta-forge files likely affected later:

- `services/data-service/src/data_service/store.py`
- `services/data-service/src/data_service/app.py`
- `services/data-service/src/data_service/routes.py`
- `services/executor/src/executor/journal.py`
- `services/executor/src/executor/state.py`
- `services/executor/src/executor/live.py`
- `services/executor/src/executor/live_target.py`
- `services/executor/src/executor/run_live.py`
- `services/executor/src/executor/run_mainnet_preflight.py`
- `services/executor/src/executor/routes.py`
- `services/executor/src/executor/run_shadow_tick.py`
- `services/executor/src/executor/run_phase_shadow_snapshot.py`
- Docker compose and deploy workflow/env examples.

Dashboard repos likely affected later:

- `quant-dashboard-collector`
  - Replace journal path inputs with DB/API inputs.
  - Add instance slug in payload generation.
- `quant-dashboard-realtime`
  - Replace equity journal path input with DB/API baseline input.
  - Add instance slug.
- `quant-dashboard-worker`
  - Allow instance-aware keys/routes.
  - Maintain current default route compatibility.
- `quant-dashboard-frontend`
  - Add instance selection only after more than one public instance is intended.

## Testing gates

Before any live-impacting rollout:

- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run ty check`
- `uv run pytest -q`
- DB migration dry-run on local Postgres.
- JSONL-to-DB import test on copied historical journals.
- DB-vs-JSONL report equivalence check.
- Dashboard collector payload contract tests.
- Mainnet preflight updated to validate DB connectivity, instance lock, and checkpoint/journal write permissions.
- One full dry-run live tick in isolated environment.
- For mainnet: explicit Boss approval, deploy once, observe next tick.

## Risks and design responses

### Risk: duplicate live runtimes trade the same account

Response:

- First-class `live_instance_id`.
- DB advisory lock or lease on startup.
- Fail closed if lock unavailable.
- Preflight should surface owner/run metadata.

### Risk: public dashboard leaks account/wallet/private fields

Response:

- Keep public payload generation as a sanitizer boundary.
- Dashboard Worker continues validating allowed top-level keys and forbidden key names.
- Do not expose `account_id`, address, quantities, order ids, raw exchange payloads, or balances in public contracts.

### Risk: DB outage stops live trading or corrupts state

Response:

- Initial dual-write mode validates DB while file remains source of truth.
- When DB becomes source of truth, live should fail closed on checkpoint/journal write failure rather than trade without persistence.
- Keep a JSON export/backup fallback during transition.

### Risk: DB schema becomes too generic / JSON-only

Response:

- Use relational columns for identifiers, timestamps, equity, drawdown, symbol, side, qty, price.
- Use JSONB only for variable diagnostics and raw compatibility fields.

### Risk: market data DB migration slows research

Response:

- Do not block local backtests on DB in early phases.
- Keep `ParquetMarketDataStore` available.
- Move live state/journals first; market bars later.

## Open decisions

1. DB hosting target:
   - Same EC2 via Docker Compose Postgres is simplest initially.
   - Managed Postgres is cleaner operationally but adds cost and networking/secrets work.
   - Recommendation for first phase: local Docker Compose Postgres on the trading host, with explicit backup/export, then revisit managed Postgres after schema stabilizes.

2. Python DB layer:
   - Lightweight `psycopg` + SQL files keeps dependencies small.
   - SQLAlchemy/Alembic gives migration ergonomics but adds framework weight.
   - Recommendation: use SQL migration files plus `psycopg`/connection pool first; add Alembic only if migration complexity grows.

3. Public multi-instance dashboard:
   - Do we show multiple public live instances, or keep only one public aggregate/default?
   - Recommendation: design instance-aware routes now, but keep frontend single-instance until Boss explicitly wants public multi-instance display.

4. Account secret storage:
   - Keep secrets in private env files initially.
   - Do not store private keys in DB.
   - DB stores only account labels, network, address prefix/hash, and runtime metadata.

## Recommended implementation and rollout plan

This section is a planning artifact only. Do not start implementation until explicitly approved.

The preferred delivery shape is a small number of coherent PRs. Avoid a permanent file+DB dual-maintenance architecture, but use temporary dual-write or shadow validation where it materially reduces live risk.

### Milestone A — freeze current behavior and add identity

Status: implementation started after approval on 2026-05-14. First slice stays intentionally small: runtime identity plumbing, additive journal metadata, compatibility tests, and docs only. No PostgreSQL dependency and no live-risk behavior change.

Progress trace:

- 2026-05-14: Added optional runtime identity fields to cta-forge live journal records and validated default backward compatibility.
- 2026-05-14: Added downstream dashboard collector/realtime tests proving the new identity metadata is ignored by public payload generation and does not truncate source numeric precision.
- 2026-05-14: Introduced small live journal/state persistence ports while keeping the existing JSONL/JSON file implementations as the only runtime backends.
- 2026-05-14: Made `LiveEngine` accept injected journal/state stores while defaulting to existing file-backed persistence.
- 2026-05-14: Added schema-only PostgreSQL migration draft for live persistence and public dashboard instance metadata; no runtime DB wiring.
- 2026-05-14: Added Decimal-safe file import helpers for existing live JSONL/state records as a foundation for historical PostgreSQL import.
- 2026-05-14: Added schema-shaped row normalization for Decimal-safe live import batches and dry-checked it against existing local journal artifacts.
- 2026-05-14: Tightened the live trade schema with an import idempotency constraint before adding DB write code.
- 2026-05-14: Added connection-protocol PostgreSQL upsert helpers for reference rows and normalized import rows, covered by fake-connection tests; still no runtime DB dependency or live wiring.
- 2026-05-14: Added a safe live persistence import CLI whose default mode is dry-run summary; DB writes require explicit `--write --database-url`.
- 2026-05-14: Extracted reusable live state payload encode/decode helpers so file and future DB checkpoint stores share one checkpoint format.

Scope:

- Add tests/fixtures for current JSONL journals, state file, dashboard public payloads, and live report conversion.
- Add `LIVE_INSTANCE_ID`, `PUBLIC_INSTANCE_SLUG`, and generated `RUN_ID` to runtime config.
- Add those identifiers to new journal records as additive fields while preserving old readers.
- Add docs/env examples showing one instance only.

Exit criteria:

- Existing live default behavior unchanged when new env vars are absent.
- Tests prove old JSONL records are still readable.
- A second dry-run instance can write to a separate path without code changes.

### Milestone B — storage ports with file-backed implementations

Scope:

- Extract `LiveJournalStore` and `LiveStateStore` ports.
- Move current `TradeJournal` and JSON state behavior behind file-backed implementations.
- Keep `LiveEngine` depending on interfaces instead of concrete files.
- Add compatibility tests proving report/dashboard data generated from the interface matches current output.

Exit criteria:

- No DB dependency yet.
- No live-risk config change.
- Code shape is ready for PostgreSQL without touching strategy logic again.

### Milestone C — PostgreSQL schema, migrations, and historical import

Scope:

- Add migration SQL for instance, run, tick, target, trade, signal, position, and checkpoint tables.
- Add local Postgres compose/profile for development only.
- Add JSONL/state import command for copied historical journals.
- Add DB-vs-JSONL equivalence checks for equity curve, drawdown, latest target, trade count, and dashboard snapshot fields.

Exit criteria:

- Historical live journals can be imported into DB reproducibly.
- DB-derived reports match file-derived reports within explicit tolerances.
- No production runtime uses DB as source of truth yet.

### Milestone D — temporary dual-write validation

Scope:

- Add `PERSISTENCE_BACKEND=file|postgres|dual`, default `file`.
- In `dual`, write file + DB for live journals/checkpoints.
- Keep reads from file initially.
- Add health/preflight checks for DB connectivity and instance lock only when DB mode requires them.

Exit criteria:

- Dry-run/shadow environment dual-writes successfully.
- DB and file summaries match after multiple ticks.
- Failures are explicit and fail closed in DB-required modes.

### Milestone E — DB read path and clean cutover

Scope:

- Switch executor live reports and dashboard collector/realtime inputs to DB/API reads for one approved instance.
- Keep JSON export for rollback during the cutover window.
- Make DB checkpoint the source of truth only after import and equivalence checks pass.

Exit criteria:

- Dashboard latest/history/realtime payloads generated from DB match current public contracts.
- Live restart restores checkpoint from DB correctly in dry-run first.
- Mainnet cutover has explicit approval, preflight evidence, and next-tick observation.

### Milestone F — dashboard multi-instance support

Scope:

- Add Worker `/instances` endpoint and instance-aware latest/history/realtime endpoints.
- Store Worker KV/Durable Object keys by `(strategy_slug, public_instance_slug)`.
- Keep current `/live/cta-forge/public/*` endpoints as default aliases.
- Add frontend instance switcher only after there is a real second public instance to show.

Exit criteria:

- Default dashboard remains backward compatible.
- Switching instance changes API target but not chart component contracts.
- Public instance metadata passes sanitizer rules and leaks no account/private data.

### Milestone G — market data DB migration, if still worth it

Scope:

- Add `MarketDataStore` abstraction after live journals/state are stable.
- Import parquet bars into `market_bars`.
- Benchmark backtest/data-service performance.
- Keep parquet available for local research unless DB clearly improves workflow.

Exit criteria:

- Research scripts are not slowed down or forced to depend on production DB.
- Any parquet-to-DB cutover is justified by operational value, not architectural neatness.

### Suggested first PR after approval

First implementation PR should probably be Milestone A only:

1. Add runtime identity config.
2. Add additive `live_instance_id` / `run_id` journal fields.
3. Add compatibility fixtures/tests.
4. Update docs/runbook examples.

Do not include PostgreSQL in the first PR. That keeps the first review low-risk and gives us a stable identity layer for both DBMS migration and multi-live support.
