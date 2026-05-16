# DBMS Dual-Write Design Gate — 2026-05-15

Status: design gate only. This document does not authorize executor deployment, restart, or enabling PostgreSQL in live runtime.

## Purpose

The next runtime-impacting storage phase should be dual-write shadow mode: file-backed persistence remains the source of truth, while PostgreSQL receives the same journal/checkpoint writes for parity observation.

The goal is evidence, not behavior change.

## Non-goals

Do not use this phase to change:

- order generation
- target construction
- risk caps
- exchange adapter behavior
- dashboard public contract
- checkpoint restore source of truth

## Proposed mode flag

Future code should use one explicit mode flag:

```text
PERSISTENCE_BACKEND=file|dual|postgres
```

Required defaults:

- unset -> `file`
- invalid value -> fail startup/preflight before trading
- production initial value -> `file`
- `dual` requires `DATABASE_URL`, `LIVE_INSTANCE_ID`, and `RUN_ID`
- `postgres` requires `DATABASE_URL`, `LIVE_INSTANCE_ID`, `RUN_ID`, and `ALLOW_POSTGRES_SOURCE_OF_TRUTH=true`
- `postgres` mode is not allowed until after a separate approval checkpoint

`services/executor/src/executor/live_persistence_runtime.py` implements this config parser/validator as prep only. It is not wired into live runtime yet. `services/executor/src/executor/run_check_live_persistence_config.py` exposes the same validation as a safe read-only preflight that prints no database URL or secret-bearing values.

## Ordering rule

For dual-write, prefer:

1. write file journal/checkpoint first
2. write PostgreSQL second
3. keep runtime reads from file

Reason: the existing file path is proven in production. A PostgreSQL write issue must not silently prevent the existing file source from staying current unless an explicit fail-closed policy is approved.

## Failure policy to decide before code

Pick one policy before enabling dual-write:

### Option A — fail-visible, keep file source of truth

- File write succeeds.
- DB write failure is logged/notified.
- Tick may continue because trading state still uses file-backed persistence.
- A parity monitor marks DB shadow invalid until repaired.

Pros: least live-trading disruption.
Cons: DB shadow can lag and must be repaired before cutover.

### Option B — fail-closed after file write

- File write succeeds.
- DB write failure raises and fails the tick.
- Requires clear retry/recovery path before enabling.

Pros: catches DB issues immediately.
Cons: DB availability can impact live loop.

Recommendation for first shadow phase: Option A, but only if DB lag is highly visible and blocks promotion to DB source of truth.

## Write coverage

Dual-write must cover all file-backed persistence surfaces:

- checkpoint save
- tick/equity journal
- position snapshots attached to ticks
- trades
- signals
- target diagnostics

Reads remain from file in this phase.

## Parity checks

After each observed tick, compare:

- latest file bar vs latest DB bar
- row counts by stream
- latest target bar/target_ts
- latest trade timestamp/idempotency
- DB-derived report shape vs JSONL-derived report shape

`services/executor/src/executor/run_check_live_persistence_parity.py` provides a read-only CLI for file-vs-DB row parity. It exits `0` on parity ok and `2` on mismatches. Use `--ignore-run-id` only when comparing a mixed historical/runtime DB history against a file snapshot whose rows intentionally use a different rehearsal run id.

Any mismatch blocks promotion.

## Configuration required before enabling

- `DATABASE_URL`
- `LIVE_INSTANCE_ID`
- `RUN_ID`
- `ACCOUNT_ID`
- strategy/profile metadata
- public instance slug/display-name if public projection is written
- backup target and restore test evidence

Do not include wallet addresses, private keys, or raw secrets in DB metadata.

## Minimal code shape

Keep the code boring:

- existing `FileLiveStateStore` and `TradeJournal` stay unchanged where possible
- existing `PostgresLiveStateStore` and `PostgresLiveJournalStore` stay connection-injected
- `services/executor/src/executor/live_persistence_dual.py` provides tested file-first `DualLiveJournalStore` and `DualLiveStateStore` wrappers, plus `parse_persistence_backend()`
- add a small factory at runtime boundary to select file/dual/postgres stores when runtime wiring is approved
- avoid broad fallback frameworks or background queues in the first version

## Required tests before any runtime deployment

- default/unset mode builds file-backed stores only
- invalid mode fails before live loop starts
- dual mode writes file then DB in order
- DB write failure behavior matches the approved policy
- file-backed read path remains used in dual mode
- no public payload exposes `live_instance_id`, `account_id`, wallet, secret/private markers, or `exchange_order_id`

## Rollback

Rollback for dual-write must be simple:

1. set `PERSISTENCE_BACKEND=file`
2. remove/ignore DB connection env from runtime
3. restart only in an approved window
4. verify next tick writes file-backed journals/checkpoint as before
5. keep DB data for comparison; do not delete it during incident response

## 2026-05-16 code-prep review

After the production historical import, the next safe step was limited to code preparation for dual-write shadow mode. This does not authorize deployment, executor restart, or enabling `PERSISTENCE_BACKEND=dual` in production.

Readiness review found the import/runtime base healthy:

- production import completed with `wrote=true`
- write-time and independent parity were `ok=true`, mismatch count 0
- executor remained file-backed with restart 0 and no post-import risk logs
- GitHub Lint/Test passed on the import evidence commit

Two code-prep gaps were fixed before considering any runtime enablement:

- the configured env name `LIVE_PERSISTENCE_SHADOW_FAILURE_POLICY` now matches runtime parsing, while the older `SHADOW_WRITE_FAILURE_POLICY` remains accepted as a compatibility alias
- live runtime store wiring now supports `file` and file-first `dual` mode, but still fails closed for `postgres` source-of-truth mode

The dual-mode runtime wiring mirrors exact file-backed journal/checkpoint records to PostgreSQL after primary file writes. This avoids false parity mismatches from independently generated DB timestamps or checkpoint `saved_at` values. File reads remain the only live read path in dual mode.

Validation for this code-prep step:

- full unit suite: `uv run pytest -q` -> 358 passed
- lint: `uv run ruff check` -> passed
- format: `uv run ruff format --check` -> passed
- targeted type check for touched live-persistence files: `uv run ty check ...` -> passed

Next runtime-impacting action, if approved separately, is a controlled production deploy/restart while keeping `PERSISTENCE_BACKEND=file`. Only after the new code is deployed and observed should `PERSISTENCE_BACKEND=dual` be enabled in a later controlled restart window.

## Hard stop

Do not proceed to DB read/source-of-truth if any of these are true:

- DB shadow misses a tick
- DB/file report parity differs
- DB write failure behavior is untested
- backup/restore is untested
- rollback has not been rehearsed
- public contract differs from current dashboard payloads
