# Live Persistence Historical Import Plan — 2026-05-15

Status: planning evidence only. This does not authorize runtime DB wiring, executor deployment, or executor restart.

## Purpose

Before PostgreSQL becomes the long-term live persistence source, all useful previous local live JSONL/state records should be classified and imported into DB. Local files should then become migration input or backup evidence, not a permanent source of truth.

This plan is intentionally conservative:

- exact duplicate copies are excluded
- older covered snapshots are excluded when their bar range and tick-time range are fully covered by a larger/later representative
- blocked or partial-overlap artifacts require review
- no records are merged automatically
- no numeric values are truncated upstream

## Command

```bash
uv run python -m executor.run_plan_live_persistence_import \
  --root live-report \
  --root backtest-results/openclaw-cleanup-20260512/workspace/artifacts
```

## Current local result

Sample run on 2026-05-15 over the roots above:

- journal directories: 7
- import candidates: 1
- requires review: 1
- excluded exact duplicates: 1
- excluded covered snapshots: 4

### Import candidate

- `backtest-results/openclaw-cleanup-20260512/workspace/artifacts/cta-forge-open-study-20260507`
  - equity records: 77
  - signals: 77
  - targets: 77
  - trades: 18
  - latest bar: 77
  - latest tick: `2026-05-07T14:03:30.137784+00:00`

### Excluded exact duplicate copy

- `backtest-results/openclaw-cleanup-20260512/workspace/artifacts/cta-forge-live-20260507T080447Z`
  - same combined content hash as `cta-forge-live-20260507T080513Z`

### Excluded covered snapshots

These are older/lower-coverage snapshots whose bar and tick-time ranges are covered by `cta-forge-open-study-20260507`:

- `backtest-results/openclaw-cleanup-20260512/workspace/artifacts/cta-forge-live-20260507T080513Z`
- `live-report/openclaw-cleanup-20260512/handoff-outbox/cta-forge-live-20260507T0508Z`
- `live-report/openclaw-cleanup-20260512/handoff-outbox/cta-forge-live-check-20260507T0634Z`
- `live-report/openclaw-cleanup-20260512/workspace/reports/cta-forge-live-records-20260506/journal`

### Requires review

- `live-report`
  - reason: duplicate bar `1` in equity/signals
  - current import shape rejects this because PostgreSQL keys ticks/signals by `(live_instance_id, bar)`
  - duplicate equity records:
    - bar 1 at `2026-04-15T18:00:32.021202+00:00`, equity `2010.75`, positions 1
    - bar 1 at `2026-04-16T00:00:32.182013+00:00`, equity `2010.82`, positions 1
  - duplicate signal records:
    - bar 1 at `2026-04-15T18:00:32.021313+00:00`
    - bar 1 at `2026-04-16T00:00:32.182873+00:00`
  - chronological equity stream otherwise spans bar 1 through 9 from `2026-04-15T18:00:32.021202+00:00` to `2026-04-18T00:00:32.319400+00:00`
  - do not import silently; decide whether this is a separate historical instance, a malformed local artifact to exclude, or a record that needs an explicit approved repair/reindex rule

## Fresh local DB rehearsal — 2026-05-15

A fresh local PostgreSQL rehearsal DB was created from scratch:

- database: `cta_forge_plan_rehearsal`
- migration: `infra/db/migrations/001_live_persistence.sql`
- table count after migration: 12
- imported only the current `import_candidate`
- did not import `live-report` because it remains `review_blocked`

Import command shape:

```bash
uv run python -m executor.run_import_live_persistence \
  --journal-dir backtest-results/openclaw-cleanup-20260512/workspace/artifacts/cta-forge-open-study-20260507 \
  --live-instance-id mainnet-pilot \
  --run-id historical-import-plan-20260515 \
  --write \
  --parity-check \
  --database-url postgresql:///cta_forge_plan_rehearsal \
  --profile-slug v16a-mainnet-pilot \
  --profile-id v16a-mainnet-pilot \
  --account-id historical-mainnet-pilot \
  --network mainnet \
  --account-label historical-mainnet-pilot \
  --mode mainnet_pilot \
  --public-instance-slug mainnet-pilot \
  --public-display-name "Mainnet Pilot" \
  --public-enabled
```

Result:

- parity: ok
- mismatch count: 0
- checkpoint rows: 0
- ticks: 77
- positions: 282
- targets: 77
- trades: 18
- signals: 77
- latest tick: bar 77 at `2026-05-07T14:03:30.137784+00:00`
- latest target: bar 77, target_ts `2026-05-07T13:00:00+00:00`

Idempotency:

- Re-running the same import kept row counts stable.

DB-derived report parity:

- `PostgresLiveJournalStore` report shape equals source `TradeJournal` JSONL report shape.
- equity records: 77
- closed trades in report: 0
- bars: 77
- latest positions: 7

Read-only file-vs-DB parity CLI:

```bash
uv run python -m executor.run_check_live_persistence_parity \
  --journal-dir backtest-results/openclaw-cleanup-20260512/workspace/artifacts/cta-forge-open-study-20260507 \
  --database-url postgresql:///cta_forge_plan_rehearsal \
  --live-instance-id mainnet-pilot \
  --run-id historical-import-plan-20260515
```

- result: `ok=true`
- mismatch count: 0
- DB/file counts matched: ticks 77, positions 282, targets 77, trades 18, signals 77
- latest tick matched bar 77 at `2026-05-07T14:03:30.137784+00:00`

Config preflight CLI smoke test:

- unset environment -> safe `backend=file`
- dual-mode env with `DATABASE_URL`, `LIVE_INSTANCE_ID`, and `RUN_ID` -> valid safe output without printing the database URL

Public-safety note:

- The importer writes public dashboard instance rows as `hidden` by default.
- The public-safe loader returned no non-hidden rows in this rehearsal, so no public payload exposed `live_instance_id`, `account_id`, wallet, secret/private markers, or `exchange_order_id`.

Packaging note:

- `run_import_live_persistence --write` requires `psycopg`; the executor package now declares `psycopg[binary]>=3.2.0` so the maintained import CLI can run DB writes in a project environment.

## Planned import runner

A review-first runner now wraps the canonical plan and existing single-journal importer:

```bash
uv run python -m executor.run_import_live_persistence_plan \
  --root live-report \
  --root backtest-results/openclaw-cleanup-20260512/workspace/artifacts \
  --live-instance-id mainnet-pilot \
  --run-id historical-import-plan-20260515
```

Dry-run result on 2026-05-15:

- `write_requested`: false
- `wrote`: false
- journal directories: 7
- import candidates: 1
- requires review: 1
- excluded exact duplicates: 1
- excluded covered snapshots: 4
- candidate rows: ticks 77, positions 282, targets 77, trades 18, signals 77, checkpoint 0

The runner keeps production writes gated:

- `--write` requires `--database-url`.
- `--write` refuses plans with review items unless `--allow-review` is supplied after explicit approval.
- `--write` currently requires exactly one import candidate, matching the current approved-candidate shape and avoiding accidental multi-artifact merges.
- write mode always enables the existing post-write parity check.

## Next rehearsal gate

Before any production DB import or runtime cutover:

1. Review/approve the candidate/exclusion plan.
2. Decide how to handle blocked `live-report`.
3. Preserve fresh-DB import evidence for approved artifacts.
4. Keep runtime source as file until a separate explicit approval checkpoint.
