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

## Production import record

2026-05-16T07:56Z approval was granted for the production DB historical import.

Before writing, the active EC2 journal was reviewed because it superseded the older local 77-tick snapshot used in the pre-approval review:

- source snapshot: `/home/admin/ops/cta-forge/import-staging/mainnet-pilot-20260516T080417Z`
- source copied from: `/home/admin/cta-forge/journal/mainnet-pilot`
- import run id: `historical-import-mainnet-pilot-20260516-active`
- pre-import backup: `/home/admin/ops/cta-forge/backups/cta_forge_live_pre_import_20260516T075902Z.dump`
- dry-run output: `/home/admin/ops/cta-forge/import-staging/outputs/dryrun-historical-import-mainnet-pilot-20260516-active-20260516T080417Z.json`
- write output: `/home/admin/ops/cta-forge/import-staging/outputs/write-historical-import-mainnet-pilot-20260516-active-20260516T080506Z.json`
- independent parity output: `/home/admin/ops/cta-forge/import-staging/outputs/parity-historical-import-mainnet-pilot-20260516-active-20260516T080555Z.json`

Production write result:

- `write_requested`: `true`
- `wrote`: `true`
- journal dirs: 1
- import candidates: 1
- requires review: 0
- ticks: 287
- positions: 1578
- targets: 287
- trades: 35
- signals: 287
- checkpoint: 0
- latest tick: bar 287 at `2026-05-16T08:03:31.868171+00:00`
- latest target: bar 287, target timestamp `2026-05-16T07:00:00+00:00`, profile `v16a-mainnet-pilot`
- latest trade in independent parity report: bar 280, `target_sell` `LINK`, `2026-05-16T02:03:35.346514+00:00`
- write-time parity: `ok=true`, mismatch count 0
- independent parity: `ok=true`, mismatch count 0

Post-import database counts:

- `live_ticks`: 287
- `live_positions`: 1578
- `live_targets`: 287
- `live_trades`: 35
- `live_signals`: 287
- `engine_checkpoints`: 0
- `live_instances`: 1
- `live_runs`: 1
- `public_dashboard_instances`: 1
- non-hidden public dashboard rows: 0

Runtime safety confirmation after import:

- no executor deploy
- no executor restart; `cta-forge-executor` still started at `2026-05-15T08:30:47.889243531Z`
- postgres healthy; restart 0; OOM false
- executor restart 0; OOM false
- runtime remains `PERSISTENCE_BACKEND=file`
- `DATABASE_URL` remains configured but not source-of-truth
- no dual-write enabled
- post-import strict mainnet preflight completed from the unchanged executor
- executor risk log grep since `2026-05-16T08:00:00Z`: 0
- postgres warning/error grep since `2026-05-16T08:00:00Z`: 0

Important correction to the pre-approval recommendation: the approved local candidate was not imported directly because the active production journal contained a newer complete file-backed record. Importing the active 287-tick snapshot was safer and more complete than importing the stale 77-tick local snapshot.

## 2026-05-17 production catch-up import record

Boss approved the production DB catch-up import at `2026-05-17T07:17Z`.

Purpose: catch PostgreSQL up from the previous historical import at bar `287` to the current file-backed `mainnet-pilot` journal snapshot, while keeping runtime file-backed. This was a PostgreSQL mutation only; it did not deploy, restart, enable dual-write, or make PostgreSQL source of truth.

Pre-import runtime state at `2026-05-17T07:20Z`:

- `cta-forge-executor` running since `2026-05-17T03:24:55.564681861Z`, restart `0`, OOM `false`.
- `cta-forge-postgres` healthy.
- Runtime backend remained `PERSISTENCE_BACKEND=file`.
- Active file journal snapshot:
  - equity/signals/targets: `310` rows each;
  - positions implied by import: `1758` rows;
  - trades: `37` rows;
  - latest tick: bar `310`, `2026-05-17T07:03:34.430302+00:00`;
  - latest target: bar `310`, target timestamp `2026-05-17T06:00:00+00:00`;
  - state file bar count: `310`.
- Pre-import DB counts remained at the prior historical import: ticks `287`, positions `1578`, targets `287`, trades `35`, signals `287`, checkpoints `0`.

Evidence paths on EC2:

- pre-catch-up backup: `/home/admin/ops/cta-forge/backups/cta_forge_live_pre_catchup_20260517T072043Z.dump`
- staged snapshot: `/home/admin/ops/cta-forge/import-staging/mainnet-pilot-catchup-20260517T072043Z`
- dry-run output: `/home/admin/ops/cta-forge/import-staging/outputs/dryrun-catchup-mainnet-pilot-20260517T072043Z.json`
- write output: `/home/admin/ops/cta-forge/import-staging/outputs/write-catchup-mainnet-pilot-20260517T072043Z.json`
- independent parity output: `/home/admin/ops/cta-forge/import-staging/outputs/parity-catchup-mainnet-pilot-20260517T072043Z.json`

Catch-up details:

- exact source path used for planning/import: staged copy of `/home/admin/cta-forge/journal/mainnet-pilot`;
- broad `/app/journal` was intentionally not used because it contains legacy/root and shadow diagnostic files;
- import run id reused: `historical-import-mainnet-pilot-20260516-active`;
- reason for reuse: `live_trades` uniqueness includes `run_id`, so a new import run id would duplicate old trade rows instead of only adding missing trades.

Dry-run result:

- import candidates: `1`
- requires review: `0`
- ticks: `310`
- positions: `1758`
- targets: `310`
- trades: `37`
- signals: `310`
- checkpoint: `0`

Write result:

- `write_requested`: `true`
- `wrote`: `true`
- write-time parity: `ok=true`, mismatch count `0`
- independent parity against the staged journal snapshot: `ok=true`, mismatch count `0`

Post-import DB counts:

- `live_ticks`: `310`, latest bar `310`
- `live_positions`: `1758`
- `live_targets`: `310`, latest bar `310`
- `live_trades`: `37`, latest bar `291`
- `live_signals`: `310`, latest bar `310`
- `engine_checkpoints`: `0`

Post-import runtime safety confirmation:

- executor start time unchanged: `2026-05-17T03:24:55.564681861Z`
- executor restart count still `0`
- executor OOM still `false`
- postgres health still `healthy`
- backend still `PERSISTENCE_BACKEND=file`
- no executor risk/error/reject/insufficient/OOM/traceback lines since the import window
- read-only mainnet preflight after import: equity `102.642868`, available `66.942649`, positions `8`, open orders `0`, target status `ok`, target timestamp `2026-05-17T06:00:00+00:00`, target orders `0`

Next boundary: enabling `PERSISTENCE_BACKEND=dual` still requires a separate controlled restart window and precheck. PostgreSQL is now caught up to the selected file-backed snapshot, but it is still not the runtime source of truth.
