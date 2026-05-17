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

## Full historical import requirement

Boss explicitly wants DBMS persistence to become complete enough that all previous useful local live records can be manually or otherwise imported into DB. After that cutover, the old local JSONL/state channel should be treated as migration input or backup evidence, not as the long-term source of truth.

Implications:

- Before DB runtime cutover, produce a local-data inventory manifest covering known journal/state artifact directories.
- Classify each artifact as import-ready, duplicate/ambiguous, malformed, or intentionally out of scope.
- Import all approved historical local records into PostgreSQL before declaring DB persistence complete.
- Preserve raw local files until DB import, parity, backup, and restore evidence are all complete.
- Do not silently drop duplicate or malformed local records; document the resolution for each blocked artifact.

## Current proven baseline

As of 2026-05-15:

- PostgreSQL schema `001_live_persistence.sql` applies cleanly to local PostgreSQL 15.
- Historical JSONL/state import is Decimal-safe.
- Local historical data must be inventoried before final DBMS cutover so previous local records can be imported into DB rather than remaining a permanent dependency.
- Import CLI is dry-run by default.
- DB writes require explicit `--write --database-url`.
- `--parity-check` reads DB rows back in the same transaction and fails closed on mismatches.
- Duplicate tick/signal bars are rejected before write because current schema keys ticks/signals by `(live_instance_id, bar)`.
- A non-duplicate 77-tick artifact imported with parity ok and idempotent row counts.
- A canonical-plan fresh DB rehearsal imported the approved candidate with parity ok and stable row counts.
- DB-derived journal read model produced the same live report shape as the source JSONL artifact.
- Read-only file-vs-DB parity CLI passed against the local rehearsal DB and restored rehearsal DB.
- PostgreSQL backup/restore runbook and local restore rehearsal evidence exist in `docs/DBMS_BACKUP_RESTORE_RUNBOOK_2026-05-15.md`.
- Live runtime remains file-backed; no DB runtime wiring exists yet.

## 2026-05-17 current phase and remaining priority

Current stage: PostgreSQL infrastructure and historical journal import are proven, and file-first dual-write code is deployed, but production runtime still runs in `PERSISTENCE_BACKEND=file` mode. The next work should stay in a shadow-validation lane until explicitly approved otherwise.

Read-only production snapshot at `2026-05-17T05:32Z`:

- executor container running; postgres container healthy;
- non-secret runtime guard still `PERSISTENCE_BACKEND=file`, `LIVE_INSTANCE_ID=mainnet-pilot`, `HL_NETWORK=mainnet`, `DRY_RUN=false`, `STRATEGY_PROFILE=v16a-mainnet-pilot`;
- active file journal latest bar `308`, latest tick `2026-05-17T05:03:34Z`;
- PostgreSQL latest imported bar `287`, from the approved `2026-05-16` historical import;
- current file-vs-DB gap is expected because dual-write is not enabled yet: ticks/signals/targets `+21`, positions `+164`, trades `+2`;
- DB checkpoint rows are `0`; file checkpoint exists at bar `308`. This is expected before dual-write checkpoint mirroring starts.

Dry-run import planning against the active instance path `/app/journal/mainnet-pilot` found exactly one import candidate with no review blockers:

- ticks `308`
- positions `1742`
- targets `308`
- trades `37`
- signals `308`
- latest target timestamp `2026-05-17T04:00:00Z`

Do not use broad root `/app/journal` for the next production catch-up import without review: it also contains legacy/root and shadow diagnostic files, and the planner classifies that broader scan as requiring review. Use the exact active instance path for the mainnet-pilot catch-up slice.

Recommended remaining order:

1. Read-only precheck and freeze point
   - verify executor/postgres health, runtime guards, open orders, latest file bars, and DB counts;
   - choose a post-tick quiet window for any write/restart action.
2. DB backup before production mutation
   - run `pg_dump` for `cta_forge_live` before any catch-up import;
   - keep the dump path in the deployment/import evidence.
3. Production DB catch-up import, journals only
   - import active `/app/journal/mainnet-pilot` into PostgreSQL with `--write` and built-in parity check;
   - reuse the existing import run id `historical-import-mainnet-pilot-20260516-active` for this catch-up slice, because `live_trades` uniqueness includes `run_id` and a new import run id would duplicate old trade rows instead of simply adding the missing trades;
   - keep runtime mode `file`; no executor restart is required for this step;
   - expect journal parity to pass after catch-up, but checkpoint parity to remain intentionally absent until dual-write is enabled.
4. Enable `PERSISTENCE_BACKEND=dual` in a controlled restart window
   - file remains source of truth;
   - PostgreSQL mirrors exact file rows/checkpoints after file writes;
   - keep `LIVE_PERSISTENCE_SHADOW_FAILURE_POLICY=warn` for the first observation window unless a stricter fail-closed policy is separately approved.
5. Observe several ticks in dual mode
   - check container restart/OOM state, risk logs, latest tick completion, open orders, and file-vs-DB parity after each tick;
   - stop immediately on any parity mismatch or shadow DB write failure.
6. Trial DB read path for reports/dashboard only
   - compare DB-derived latest/history payloads against the existing public API contract;
   - grep public payloads for private identifiers before exposing anything;
   - trading runtime still reads file state in this phase.
7. Separate middleware lifecycle from routine executor deploy
   - track with `rocky-d/cta-forge#1`;
   - split DB infra/migration operations from ordinary executor deploys before the DB becomes a long-term critical runtime dependency.
8. DB checkpoint source-of-truth cutover
   - highest-risk storage phase;
   - only after dual-write parity is stable and rollback is rehearsed;
   - requires separate explicit approval.
9. True multi-live activation
   - define the next `LIVE_INSTANCE_ID`, account identity, risk caps, public slug policy, dashboard routing, and deployment isolation;
   - do not start a second real live instance until DB identity/isolation and dashboard/report semantics are proven with `mainnet-pilot`.

## Local inventory command

Use the inventory command before any full historical DB import:

```bash
uv run python -m executor.run_inventory_live_persistence \
  --root live-report \
  --root backtest-results/openclaw-cleanup-20260512/workspace/artifacts
```

The command emits a JSON manifest with:

- discovered journal directories
- equity/trade/signal/target row counts
- first/latest tick summaries
- latest target summary
- per-file and combined content hashes
- duplicate-content groups across archival copies
- cross-directory bar overlaps that need canonical-source selection before DB import
- candidate `engine-state.json` files
- duplicate tick/signal bars that block the current PostgreSQL import shape
- parse errors for malformed JSONL

Use `--fail-on-blocked` in automation when all discovered directories are expected to be import-ready.

First local inventory sample on 2026-05-15 over `live-report` plus recovered artifact roots found 7 journal directories: 6 ready and 1 blocked due duplicate bar `1` in `live-report` equity/signals. The inventory also reports duplicate archival copies and overlapping bar ranges so final import can pick canonical segments instead of blindly re-importing every copied directory.

## Local canonical import-plan command

After inventory, build a conservative canonical import plan:

```bash
uv run python -m executor.run_plan_live_persistence_import \
  --root live-report \
  --root backtest-results/openclaw-cleanup-20260512/workspace/artifacts
```

The plan classifies artifacts as:

- `import_candidate`
- `exclude_exact_duplicate`
- `exclude_covered_snapshot`
- `review_blocked`
- `review_overlap_identity`

Use `--fail-on-review` in automation when no blocked or ambiguous artifacts are expected. Current planning and fresh local DB rehearsal evidence lives in `docs/LIVE_PERSISTENCE_IMPORT_PLAN_2026-05-15.md`.

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
- [ ] Local-data inventory manifest is generated and reviewed.
- [x] Canonical import plan is generated and reviewed (`docs/LIVE_PERSISTENCE_IMPORT_REVIEW_2026-05-16.md`).
- [x] Every discovered local journal/state artifact is classified as import candidate, duplicate/covered exclusion, blocked, or intentionally out of scope for the May 2026 mainnet-pilot import.
- [x] Historical import from copied production journals passed production `--write` with parity check for the 2026-05-16 active `mainnet-pilot` snapshot (`docs/LIVE_PERSISTENCE_IMPORT_REVIEW_2026-05-16.md`).
- [x] DB-derived import rows matched JSONL-derived rows for the copied active production journal; independent parity `ok=true`, mismatch count 0.
- [ ] Duplicate-bar or other ambiguous historical records are either fixed by an approved migration rule or explicitly excluded with evidence.
- [x] Backup and restore have been tested on non-production DB (`cta_forge_plan_restore_rehearsal_20260515`, parity ok).
- [ ] Public dashboard payload grep shows no private identifiers.

## Proposed production rollout sequence

### Phase 0 — production DB prepare, no runtime changes

1. Provision PostgreSQL.
2. Create DB/user with least necessary permissions.
3. Apply migrations.
4. Run schema verification queries.
5. Take initial empty DB backup.
6. Stop. No executor changes.

### Phase 1 — full historical import rehearsal from local/production copies

1. Generate an inventory manifest for known local journal/state roots.
2. Classify each discovered artifact as import-ready, duplicate/ambiguous, malformed, or intentionally out of scope.
3. Copy production journals/state to a safe staging path when needed.
4. Run import CLI dry-run for each approved artifact.
5. Run import CLI with `--write --parity-check` against staging/test DB.
6. Compare DB-derived report to JSONL-derived report for representative artifacts.
7. Record row counts and latest tick/target/trade evidence.
8. Stop. No executor changes.

### Phase 2 — dual-write shadow mode, file remains source of truth

Requires new code and separate approval. Design gate: `docs/DBMS_DUAL_WRITE_DESIGN_2026-05-15.md`.

1. [x] Add `PERSISTENCE_BACKEND=file|dual|postgres`, default `file`.
2. [x] In `dual`, live runtime writes file first and mirrors exact file records to DB second.
3. [x] Reads remain from file in the prepared code path.
4. [x] DB failures are visible and follow `LIVE_PERSISTENCE_SHADOW_FAILURE_POLICY=warn|raise`.
5. [x] Deploy/restart the prepared code while keeping production `PERSISTENCE_BACKEND=file`.
6. [ ] Enable `PERSISTENCE_BACKEND=dual` only in a later approved restart window.
7. [ ] Observe multiple ticks.
8. [ ] Compare file and DB after each tick.
9. [ ] Stop if any parity mismatch occurs.

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

## 2026-05-16 code-only dual-write preparation deployment

After the production historical import, commit `4b3e892` was deployed with no runtime backend change.

Deployment boundary:

- GitHub Deploy run: `25958222641`
- deployed commit: `4b3e892 feat(executor): prepare file-first live persistence dual write`
- purpose: ship file-first dual-write preparation code only
- production runtime mode remained `PERSISTENCE_BACKEND=file`
- dual-write was not enabled
- PostgreSQL source-of-truth was not enabled

Pre-deploy check after the 09:00 UTC tick:

- executor running since `2026-05-15T08:30:47.889243531Z`, restart 0, OOM false
- postgres running and healthy, restart 0, OOM false
- env guards included `HL_NETWORK=mainnet`, `DRY_RUN=false`, `STRATEGY_PROFILE=v16a-mainnet-pilot`, `PERSISTENCE_BACKEND=file`, `LIVE_INSTANCE_ID=mainnet-pilot`
- active file journal latest: equity/signals/targets bar 288; trades 36 with latest bar 287 `target_sell` `ARB`
- DB counts remained at imported bar 287: ticks 287, positions 1578, targets 287, trades 35, signals 287, checkpoints 0
- executor risk log grep since `2026-05-16T09:00:00Z`: 0

Immediate post-deploy check:

- executor restarted at `2026-05-16T09:18:50.241877311Z`, restart 0, OOM false
- postgres remained healthy, restart 0, OOM false
- runtime guard still `PERSISTENCE_BACKEND=file`; `DATABASE_URL` configured only for prepared tooling
- startup log printed live persistence config with `backend: file` and `allow_postgres_source_of_truth: false`
- startup preflight passed and the engine waited for the 10:00 UTC candle
- strict mainnet preflight after deploy passed: equity `103.883919`, available balance `80.857696`, open orders 0, positions ARB/BNB/BTC/DOGE/INJ/SEI/SOL, target status ok, target gross `0.8728350732337687`, target timestamp `2026-05-16T08:00:00+00:00`, target orders 0
- file journal stayed at bar 288 until the next tick; DB counts stayed unchanged at the imported bar 287, as expected for file mode
- executor startup/risk grep since container start: 0
- postgres warning/error grep since deploy start: 0

Next approval boundary: observe the first post-deploy tick, then separately approve enabling `PERSISTENCE_BACKEND=dual` in a controlled restart window if file-mode behavior remains clean.

## 2026-05-15 containerized PostgreSQL deployment note

Boss clarified that production PostgreSQL should be deployed containerized on EC2 rather than as a host package. The current implementation direction is documented in `docs/DBMS_CONTAINER_DEPLOYMENT_2026-05-15.md`:

- `postgres:17-alpine` managed by `docker-compose.prod.yml`;
- named volume `cta_forge_postgres_data` for data;
- private EC2 `.env` for current practical secrets;
- GitHub Actions starts PostgreSQL, applies migrations with a one-shot `db-migrate` service, then recreates only the executor;
- live runtime remains `PERSISTENCE_BACKEND=file` until a separate dual-write approval gate.

## 2026-05-17 dual-write run registration precheck

After the production catch-up import, a code-level precheck found one more required safety detail before enabling `PERSISTENCE_BACKEND=dual`: runtime-generated `RUN_ID` values must exist in `live_runs` before any shadow journal/checkpoint row is written. Otherwise PostgreSQL foreign keys can reject dual-mode writes even though the historical import rows are valid.

Fix direction implemented in code prep: dual-mode store initialization now registers the current runtime run id in `live_runs` with non-secret runtime persistence metadata before constructing the PostgreSQL mirror stores. The existing `live_instance_id` must already exist from the approved import/cutover setup; if not, startup fails closed.

This remains code preparation only. Production dual-write still requires a separate controlled restart/config-change approval.
