# DBMS Backup / Restore Runbook — 2026-05-15

Status: PostgreSQL rehearsal/runbook only. This does not authorize runtime DB wiring, executor deployment, or source-of-truth cutover.

## Purpose

Before PostgreSQL can become the long-term live persistence source, backup and restore must be boring, rehearsed, and verifiable.

## Minimum rule

A database is not a source of truth until a restore has been tested and the restored database passes read-model/parity checks.

## Backup command shape

Use a custom-format dump for portable restore tests:

```bash
pg_dump -Fc -f /secure/backup/path/cta_forge_live_$(date -u +%Y%m%dT%H%M%SZ).dump "$DATABASE_URL"
```

Operational notes:

- store backups outside the application container when running in production
- restrict file permissions on backup artifacts
- do not print `DATABASE_URL` or secret-bearing connection strings in logs
- keep at least one backup from before any schema migration
- keep local JSONL/state artifacts as migration evidence until DB promotion is fully complete and separately approved

## Restore command shape

Restore into a fresh database first. Do not restore over a live database during normal verification.

```bash
createdb cta_forge_restore_rehearsal
pg_restore -d cta_forge_restore_rehearsal /secure/backup/path/cta_forge_live.dump
```

Then run read-only verification:

```bash
uv run python -m executor.run_check_live_persistence_parity \
  --journal-dir <canonical-file-journal-dir> \
  --database-url postgresql:///cta_forge_restore_rehearsal \
  --live-instance-id <live-instance-id> \
  --run-id <run-id>
```

Expected result before promotion:

- `ok=true`
- mismatch count 0
- latest tick/target/trade/signal match the canonical evidence
- public rows stay hidden unless explicitly made public through approved dashboard flow

## Local restore rehearsal — 2026-05-15

Source database:

- `cta_forge_plan_rehearsal`

Backup:

```bash
pg_dump -Fc -f /tmp/cta_forge_plan_rehearsal.dump cta_forge_plan_rehearsal
```

Restore target:

- `cta_forge_plan_restore_rehearsal_20260515`

Restore:

```bash
createdb cta_forge_plan_restore_rehearsal_20260515
pg_restore -d cta_forge_plan_restore_rehearsal_20260515 /tmp/cta_forge_plan_rehearsal.dump
```

Verification command:

```bash
uv run python -m executor.run_check_live_persistence_parity \
  --journal-dir backtest-results/openclaw-cleanup-20260512/workspace/artifacts/cta-forge-open-study-20260507 \
  --database-url postgresql:///cta_forge_plan_restore_rehearsal_20260515 \
  --live-instance-id mainnet-pilot \
  --run-id historical-import-plan-20260515
```

Result:

- `ok=true`
- mismatch count: 0
- restored DB/file counts matched:
  - ticks: 77
  - positions: 282
  - targets: 77
  - trades: 18
  - signals: 77
- latest tick matched bar 77 at `2026-05-07T14:03:30.137784+00:00`

## Promotion gate

Do not promote PostgreSQL to runtime source of truth unless all are true:

- backup target is outside the runtime container
- restore rehearsal passes against a recent backup
- restore verification command and expected identifiers are documented
- rollback path is documented and rehearsed
- DB/file parity remains clean during dual-write shadow mode
