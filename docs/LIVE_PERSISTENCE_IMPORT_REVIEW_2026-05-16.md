# Live persistence import review — 2026-05-16

Status: pre-production-write review only. This document does not authorize production DB import, dual-write, source-of-truth cutover, executor deploy, or runtime restart.

## Commands reviewed

Planned import runner dry-run:

```bash
uv run python -m executor.run_import_live_persistence_plan \
  --root live-report \
  --root backtest-results/openclaw-cleanup-20260512/workspace/artifacts \
  --live-instance-id mainnet-pilot \
  --run-id historical-import-plan-20260516
```

Full plan inspection:

```bash
uv run python -m executor.run_plan_live_persistence_import \
  --root live-report \
  --root backtest-results/openclaw-cleanup-20260512/workspace/artifacts
```

## Dry-run result

- `write_requested`: `false`
- `wrote`: `false`
- journal directories discovered: 7
- import candidates: 1
- requires review: 1
- excluded exact duplicates: 1
- excluded covered snapshots: 4

## Recommended production import candidate

Import only this candidate into the `mainnet-pilot` live instance:

- journal: `backtest-results/openclaw-cleanup-20260512/workspace/artifacts/cta-forge-open-study-20260507`
- reason: largest/latest representative for the covered snapshot group
- first tick: bar 1 at `2026-05-04T10:00:37.028294+00:00`, equity `99.7`, positions `0`
- latest tick: bar 77 at `2026-05-07T14:03:30.137784+00:00`, equity `99.36`, positions `7`
- latest target: bar 77, profile `v16a-mainnet-pilot`, target timestamp `2026-05-07T13:00:00+00:00`, normalized gross `1.627801`
- row summary for planned import:
  - ticks: 77
  - positions: 282
  - targets: 77
  - trades: 18
  - signals: 77
  - checkpoint: 0

Rationale: this candidate covers the copied May 2026 mainnet-pilot snapshots and has already passed previous local fresh-DB import/parity rehearsal. It matches the intended `mainnet-pilot` account scale and profile shape.

## Exclusions reviewed

These artifacts should not be separately imported into `mainnet-pilot` because they are duplicate or fully covered by the recommended candidate:

- `backtest-results/openclaw-cleanup-20260512/workspace/artifacts/cta-forge-live-20260507T080447Z`
  - action: `exclude_exact_duplicate`
  - representative: `backtest-results/openclaw-cleanup-20260512/workspace/artifacts/cta-forge-live-20260507T080513Z`
  - counts: equity 71, signals 71, targets 71, trades 18
- `backtest-results/openclaw-cleanup-20260512/workspace/artifacts/cta-forge-live-20260507T080513Z`
  - action: `exclude_covered_snapshot`
  - representative: recommended candidate
  - counts: equity 71, signals 71, targets 71, trades 18
- `live-report/openclaw-cleanup-20260512/handoff-outbox/cta-forge-live-20260507T0508Z`
  - action: `exclude_covered_snapshot`
  - representative: recommended candidate
  - counts: equity 68, signals 68, targets 68, trades 17
- `live-report/openclaw-cleanup-20260512/handoff-outbox/cta-forge-live-check-20260507T0634Z`
  - action: `exclude_covered_snapshot`
  - representative: recommended candidate
  - counts: equity 69, signals 69, targets 69, trades 17
- `live-report/openclaw-cleanup-20260512/workspace/reports/cta-forge-live-records-20260506/journal`
  - action: `exclude_covered_snapshot`
  - representative: recommended candidate
  - counts: equity 46, signals 46, targets 46, trades 10

## Blocked legacy artifact

The root `live-report` artifact remains blocked:

- action: `review_blocked`
- reason: duplicate bar `1` in equity and signals
- counts: equity 10, signals 10, targets 0, trades 7
- first tick: `2026-04-15T18:00:32.021202+00:00`, equity `2010.75`, positions `1`
- latest tick: `2026-04-18T00:00:32.319400+00:00`, equity `2113.57`, positions `5`
- latest target: none
- state candidate: `engine-state.json`

Review decision: do not import this artifact into `mainnet-pilot` during the May 2026 historical import. It appears to represent an older/different live context with different account scale, no target journal records, and duplicate bars that cannot be represented safely under the current `(live_instance_id, bar)` tick/signal keys.

If this data is later considered useful, handle it as a separate legacy import design with its own `live_instance_id`, duplicate-bar reindexing policy, and explicit evidence. Do not bypass it silently.

## Production write gate

Because the plan contains one blocked review item, the runner intentionally fails closed unless production write is run with explicit review acknowledgement:

```bash
uv run python -m executor.run_import_live_persistence_plan \
  --root live-report \
  --root backtest-results/openclaw-cleanup-20260512/workspace/artifacts \
  --live-instance-id mainnet-pilot \
  --run-id historical-import-mainnet-pilot-20260516 \
  --database-url "$DATABASE_URL" \
  --write \
  --allow-review
```

Use `--allow-review` only to acknowledge the documented decision above: import the single May 2026 candidate and leave root `live-report` out of this production import.

## Required checks immediately after any approved production import

- Confirm runner output has `wrote=true` and parity mismatch count `0`.
- Confirm row counts match the dry-run candidate: ticks 77, positions 282, targets 77, trades 18, signals 77.
- Confirm the latest imported tick is bar 77 at `2026-05-07T14:03:30.137784+00:00`.
- Confirm no runtime mode changed: `PERSISTENCE_BACKEND=file`.
- Confirm executor was not restarted as part of the import.
- Confirm PostgreSQL remains healthy.
- Do not enable dual-write or DB source-of-truth in the same step.

## Current recommendation

Next safe step is a separate, explicit production DB import approval using the command shape above. This is a DB mutation, not an executor deployment, and should still be treated as its own approval boundary.
