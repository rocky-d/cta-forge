# Mainnet pilot runbook — 2026-05-04

Purpose: validate real Hyperliquid mainnet connectivity, order flow, state, journal, and notifications with tiny capital before any scale-up. The first phase is operational validation, not profit seeking.

## Current frozen testnet baseline

- EC2 host: `admin@35.74.148.197`.
- Testnet freeze archive: `/home/node/references/environment/cta-forge/mainnet-cutover/testnet-freeze-20260504T072348Z.tar.gz`.
- Freeze summary: `/home/node/references/environment/cta-forge/mainnet-cutover/testnet-freeze-20260504T072348Z-summary.txt`.
- At freeze time: container running, restart count 0, OOMKilled false, `HL_NETWORK=testnet`, `DRY_RUN=false`, `STRATEGY_PROFILE=v16a-badscore-overlay`, two testnet positions, zero known stale open orders.

## Guardrail design

Mainnet v16a is not enabled by the existing testnet profile. It uses a distinct pilot profile:

- `STRATEGY_PROFILE=v16a-mainnet-pilot`
- `HL_NETWORK=mainnet`
- dry-run phase: `DRY_RUN=true`, `ALLOW_MAINNET_PILOT_LIVE=false`
- live phase: `DRY_RUN=false`, `ALLOW_MAINNET_PILOT_LIVE=true`
- state path: `/app/state/engine-state-mainnet-pilot.json`
- journal path: `/app/journal/mainnet-pilot`

Current live-pilot risk defaults for the roughly 100 USDC pilot wallet:

- `TARGET_SCALE=5.0` (scales v16a target weights so small-account orders clear Hyperliquid minimum notional)
- `TARGET_GROSS_CAP=4.00`
- `MIN_EQUITY=50` (pilot allows slightly below 100 USDC after transfer fees)
- dry-run/pre-entry `MIN_AVAILABLE_BALANCE=50`; live overlay uses `MIN_AVAILABLE_BALANCE=0` after positions exist because margin usage can legitimately reduce available balance
- `MAX_EQUITY=200`
- `MAX_ORDER_NOTIONAL=50`
- `HL_LEVERAGE=5`
- `V16A_CORE_PHASE_HOURS=0`
- `LIVE_SYMBOLS=BTC,ETH,SOL,BNB,XRP,DOGE,AVAX,LINK,ADA,DOT,ATOM,NEAR,APT,ARB,OP,SUI,INJ,TIA,SEI`

These values are intentionally both configuration defaults and code-level pilot
caps. For mainnet non-dry-run, the executable fails closed unless the guarded
`v16a-mainnet-pilot` profile is used and the pilot caps stay within:

- `MAX_EQUITY <= 200`
- `MAX_ORDER_NOTIONAL <= 50`, or empty/unset only with explicit `ALLOW_MAINNET_PILOT_UNCAPPED_ORDERS=true`
- `TARGET_GROSS_CAP <= 4.00`
- `HL_LEVERAGE <= 5`

Do not raise or remove these values from the EC2 host or compose files as an ad hoc tweak.
Any increase or no-max change is a new live-risk decision and should go through research, code
review, CI, deploy, and post-deploy health checks.

## Hyperliquid unified-account balance accounting

Current mainnet pilot wallets are unified accounts. For these accounts,
Hyperliquid's account-level web value/portfolio account value matches the USDC
`total` from `spot_user_state`, while per-dex `marginSummary.accountValue`
corresponds only to the perp-dex slice and must not be mixed with spot
available balance as account equity.

Use these pilot accounting meanings:

- account equity / web USDC value: spot USDC `total`
- available balance: spot USDC `total - hold`
- margin used: spot USDC `hold`
- per-position unrealized PnL: explanatory only; do not add it again to spot
  USDC `total`

The tick notification should therefore label `Eq`, `Avail`, and `uPnL`
separately instead of publishing one ambiguous dollar value.

## Read-only preflight

Run only after mainnet secrets are available in environment:

```bash
HL_NETWORK=mainnet \
DRY_RUN=false \
STRATEGY_PROFILE=v16a-mainnet-pilot \
ALLOW_MAINNET_PILOT_LIVE=true \
TARGET_SCALE=5.0 \
TARGET_GROSS_CAP=4.00 \
MIN_EQUITY=50 \
MIN_AVAILABLE_BALANCE=0 \
MAX_EQUITY=200 \
MAX_ORDER_NOTIONAL=50 \
HL_LEVERAGE=5 \
LIVE_SYMBOLS=BTC,ETH,SOL,BNB,XRP,DOGE,AVAX,LINK,ADA,DOT,ATOM,NEAR,APT,ARB,OP,SUI,INJ,TIA,SEI \
uv run python -m executor.run_mainnet_preflight
```

The command performs no exchange writes. It checks mainnet account state, open orders, symbol metadata, L2 snapshots, estimated minimum rounded order sizes, runtime path writability, pilot cap validity, and current pilot target diagnostics. In deploy mode it is strict: path, cap, symbol, or target-construction errors return non-zero and must block deployment.

## Deployment path

Prefer GitHub Actions CI/CD. Avoid ad hoc EC2 mutation except read-only checks and emergency stop/diagnostics.

For live pilot deployment, prefer the Deploy workflow with `target=mainnet-pilot-live`. It builds the executor image, syncs `docker-compose.prod.yml`, `docker-compose.mainnet-pilot.yml`, and `docker-compose.mainnet-pilot-live.yml`, runs strict pre-start mainnet pilot preflight, recreates the container, verifies the resulting container env includes `HL_NETWORK=mainnet`, `DRY_RUN=false`, `ALLOW_MAINNET_PILOT_LIVE=true`, and `STRATEGY_PROFILE=v16a-mainnet-pilot`, then runs strict post-start preflight again.

Before triggering a live deploy:

1. Confirm the intended commit is pushed and GitHub Lint/Test passed for that exact SHA.
2. Confirm the workflow target is `mainnet-pilot-live`; the default target is not sufficient for mainnet pilot validation.
3. Confirm the intended live env still has `V16A_CORE_PHASE_HOURS=0`, no `V16A_COMPARE_CORE_PHASE_HOURS`, and `MAX_ORDER_NOTIONAL=50` unless an explicit live-change decision has been made.
4. Prefer one deployment at a time; wait for GitHub Deploy to complete and inspect EC2 health before running research snapshots or further diagnostics.

Equivalent manual compose command, for emergency/operator use only:

```bash
docker compose \
  -f docker-compose.prod.yml \
  -f docker-compose.mainnet-pilot.yml \
  -f docker-compose.mainnet-pilot-live.yml \
  up -d
```

The prior dry-run phase uses `target=mainnet-pilot-dry-run` with only `docker-compose.mainnet-pilot.yml`, `DRY_RUN=true`, and `ALLOW_MAINNET_PILOT_LIVE=false`. Do not use that target when validating real order execution.

## Post-deploy health checklist

After every mainnet-pilot-live deploy, check at minimum:

```bash
ssh -i /home/node/.ssh/cta-forge.pem -o BatchMode=yes admin@35.74.148.197
cd ~/cta-forge
docker inspect cta-forge-executor --format 'status={{.State.Status}} restart={{.RestartCount}} oom={{.State.OOMKilled}} started={{.State.StartedAt}}'
docker inspect cta-forge-executor --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep -E '^(HL_NETWORK|DRY_RUN|ALLOW_MAINNET_PILOT_LIVE|STRATEGY_PROFILE|V16A_CORE_PHASE_HOURS|V16A_COMPARE_CORE_PHASE_HOURS|MAX_ORDER_NOTIONAL|TARGET_SCALE|TARGET_GROSS_CAP|HL_LEVERAGE)=' \
  | sort
docker logs --since 15m cta-forge-executor 2>&1 \
  | grep -Ei 'ERROR|Traceback|Exception|OOM|insufficient|reject|failed|placed' || true
```

Healthy baseline after the 2026-05-06 deployment was: container `running`,
`restart=0`, `oom=false`, strict preflight passed, no stale open orders, state
restored, and no unexpected order/reject/error logs. A later post-candle check
should verify the next scheduled tick completed normally.

## Read-only forward shadow diagnostics

Forward shadow is allowed only as a separate read-only subprocess. It must not
change the live container environment, restart the live engine, run a live tick,
refresh Binance data, or submit orders.

Use cache-only snapshots for phase/cap evidence:

```bash
# Run inside the EC2 container via docker exec. Override only the subprocess env.
docker exec \
  -e DRY_RUN=true \
  -e STRATEGY_PROFILE=v16a-badscore-overlay \
  -e V16A_COMPARE_CORE_PHASE_HOURS=2 \
  -e MAX_ORDER_NOTIONAL=50 \
  -e PHASE_COMPARISON_JOURNAL_DIR=/app/journal/mainnet-pilot-phase-shadow-cap50 \
  cta-forge-executor uv run python -m executor.run_phase_shadow_snapshot

docker exec \
  -e DRY_RUN=true \
  -e STRATEGY_PROFILE=v16a-badscore-overlay \
  -e V16A_COMPARE_CORE_PHASE_HOURS=2 \
  -e MAX_ORDER_NOTIONAL=75 \
  -e PHASE_COMPARISON_JOURNAL_DIR=/app/journal/mainnet-pilot-phase-shadow-cap75 \
  cta-forge-executor uv run python -m executor.run_phase_shadow_snapshot
```

Interpret the resulting journals as a 2x2 diagnostic matrix:

- phase `0`, cap `$50`: current baseline;
- phase `2`, cap `$50`: phase candidate only;
- phase `0`, cap `$75`: execution-cap candidate only;
- phase `2`, cap `$75`: combined candidate, research-only.

Operational notes from the first EC2 snapshot:

- The live engine uses `STRATEGY_PROFILE=v16a-mainnet-pilot`; the cache-only
  snapshot subprocess intentionally overrides to `v16a-badscore-overlay` because
  the snapshot loader currently accepts the underlying v16a target profile.
- The live container env must still show `V16A_CORE_PHASE_HOURS=0`, no
  `V16A_COMPARE_CORE_PHASE_HOURS`, and `MAX_ORDER_NOTIONAL=50` after snapshots.
- Run long snapshots one at a time with enough timeout, then confirm no stale
  `run_phase_shadow_snapshot` process remains via `docker top`.
- A `$75` shadow snapshot is not a live config change. It only tests hypothetical
  order construction. Reduce-only orders are intentionally uncapped, so `$50` vs
  `$75` matters only when the candidate adds or increases exposure.

## Stop / rollback

Stop submitting new orders:

```bash
cd ~/cta-forge
docker compose \
  -f docker-compose.prod.yml \
  -f docker-compose.mainnet-pilot.yml \
  -f docker-compose.mainnet-pilot-live.yml \
  stop -t 30 executor-live
```

Cancel open orders separately if needed. Flattening positions is a separate decision and should not be automatic unless there is an immediate risk reason.
