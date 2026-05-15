# DBMS Cutover Checklist — 2026-05-15

Status: planning/runbook only. Do not use this as approval to deploy or wire live runtime to PostgreSQL.

## Scope

This checklist covers the future cutover from file-backed live persistence to PostgreSQL-backed live persistence for cta-forge live state and journals.

It does not authorize:

- changing live order logic
- changing risk caps
- changing strategy targets
- deploying executor
- restarting production runtime
- enabling PostgreSQL as runtime source of truth

Those actions require a separate explicit approval checkpoint.

## Current proven baseline

As of 2026-05-15:

- PostgreSQL schema `001_live_persistence.sql` applies cleanly to local PostgreSQL 15.
- Historical JSONL/state import is Decimal-safe.
- Import CLI is dry-run by default.
- DB writes require explicit `--write --database-url`.
- `--parity-check` reads DB rows back in the same transaction and fails closed on mismatches.
- Duplicate tick/signal bars are rejected before write because current schema keys ticks/signals by `(live_instance_id, bar)`.
- A non-duplicate 77-tick artifact imported with parity ok and idempotent row counts.
- DB-derived journal read model produced the same live report shape as the source JSONL artifact.
- Live runtime remains file-backed; no DB runtime wiring exists yet.

## Required decisions before production cutover

1. Production PostgreSQL location
   - Option A: same EC2 host, local Docker Compose PostgreSQL.
   - Option B: managed PostgreSQL.
   - Recommendation for first production rehearsal: same host or isolated private network DB, smallest operational surface.

2. Backup target
   - Define `pg_dump` destination.
   - Define retention.
   - Define restore test path.

3. Runtime mode progression
   - `file` remains default.
   - `dual` can be introduced for shadow validation.
   - `postgres` only after dual-write parity passes.

4. Instance identity
   - Confirm `LIVE_INSTANCE_ID`.
   - Confirm `RUN_ID` generation policy.
   - Confirm `ACCOUNT_ID` and public slug mapping.
   - Confirm no public payload includes account/wallet/order identifiers.

5. Rollback owner and trigger
   - Define who can approve rollback.
   - Define rollback thresholds: DB write error, parity mismatch, missing checkpoint, dashboard mismatch, preflight failure.

## Pre-cutover evidence gate

Before any production runtime wiring:

- [ ] Local tests pass: `uv run pytest -q`.
- [ ] Ruff passes: `uv run ruff check && uv run ruff format --check`.
- [ ] Targeted ty passes for touched files.
- [ ] GitHub Lint/Test passes on pushed commit.
- [ ] Production DB migration has been rehearsed on a clean test DB.
- [ ] Historical import from copied production journals passes `--write --parity-check` on test DB.
- [ ] DB-derived live report equals JSONL-derived live report for the copied journals.
- [ ] Duplicate-bar or other ambiguous historical records are either fixed by an approved migration rule or explicitly excluded with evidence.
- [ ] Backup and restore have been tested on non-production DB.
- [ ] Public dashboard payload grep shows no private identifiers.

## Proposed production rollout sequence

### Phase 0 — production DB prepare, no runtime changes

1. Provision PostgreSQL.
2. Create DB/user with least necessary permissions.
3. Apply migrations.
4. Run schema verification queries.
5. Take initial empty DB backup.
6. Stop. No executor changes.

### Phase 1 — historical import rehearsal from production copies

1. Copy production journals/state to a safe staging path.
2. Run import CLI dry-run.
3. Run import CLI with `--write --parity-check` against staging/test DB.
4. Compare DB-derived report to JSONL-derived report.
5. Record row counts and latest tick/target/trade evidence.
6. Stop. No executor changes.

### Phase 2 — dual-write shadow mode, file remains source of truth

Requires new code and separate approval.

1. Add `PERSISTENCE_BACKEND=file|dual|postgres`, default `file`.
2. In `dual`, live runtime writes file first and DB second, or uses an explicitly chosen safe ordering.
3. Reads remain from file.
4. DB failures must be visible and fail according to approved policy.
5. Observe multiple ticks.
6. Compare file and DB after each tick.
7. Stop if any parity mismatch occurs.

### Phase 3 — DB read path for reports/dashboard only

Requires separate approval.

1. Keep trading state source as file.
2. Allow report/dashboard side to read DB-derived compatible views.
3. Verify public payload equality against existing dashboard payloads.
4. Keep strategy-level public API backward compatible.

### Phase 4 — DB checkpoint source of truth

Highest-risk storage phase; requires explicit approval.

1. Import latest file checkpoint into DB.
2. Run read-only DB checkpoint load preflight.
3. Restart only in controlled window.
4. Confirm runtime restores exact expected state.
5. Observe next tick before declaring success.
6. Keep file export fallback enabled during rollback window.

## Rollback plan

Rollback must be simple:

1. Set runtime persistence mode back to `file`.
2. Restore original `STATE_FILE` and `JOURNAL_DIR` paths.
3. Restart only if required by the approved rollback procedure.
4. Confirm preflight and next tick use file-backed state.
5. Do not delete DB data; keep it for postmortem comparison.

If rollback is due to DB corruption or wrong import:

- stop DB writes
- take a DB snapshot before touching data
- compare against file journals
- decide whether to truncate/reimport only after evidence review

## Post-cutover observation checklist

After any approved runtime-impacting phase:

- [ ] Container running, restart count stable, no OOM.
- [ ] Environment flags match approved mode.
- [ ] Strict preflight passes.
- [ ] Latest tick completes.
- [ ] Journal/checkpoint write success is visible.
- [ ] DB/file parity is checked where applicable.
- [ ] Dashboard latest/history/realtime unchanged in public contract.
- [ ] Public payload grep shows no private identifiers.
- [ ] No order reject/insufficient balance/unexpected position anomaly.

## Hard stop conditions

Stop and do not proceed to the next phase if any of these occur:

- Migration is not repeatable on clean DB.
- Import parity fails without an understood and documented reason.
- DB-derived report differs from JSONL-derived report.
- Public payload contains private identifiers.
- Runtime preflight fails.
- DB outage behavior is unclear.
- Rollback has not been rehearsed.

## Notes

- Do not introduce TimescaleDB until native PostgreSQL is proven insufficient.
- Do not store private keys in DB.
- Keep public instance metadata separate from internal account/runtime identity.
- Prefer one reversible phase at a time over a large storage cutover.
