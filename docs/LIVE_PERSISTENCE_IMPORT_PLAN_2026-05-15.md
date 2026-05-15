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
  - do not import silently; decide whether this is a separate historical instance, a malformed local artifact to exclude, or a record that needs an explicit approved repair rule

## Next rehearsal gate

Before any production DB import or runtime cutover:

1. Review/approve the candidate/exclusion plan.
2. Decide how to handle blocked `live-report`.
3. Rehearse import into a fresh local PostgreSQL DB.
4. Run `--write --parity-check` for approved artifacts.
5. Confirm DB-derived report/public read model parity.
6. Keep runtime source as file until a separate explicit approval checkpoint.
