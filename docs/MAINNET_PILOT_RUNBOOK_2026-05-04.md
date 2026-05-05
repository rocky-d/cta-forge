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
- `DRY_RUN=true` initially
- live orders require `ALLOW_MAINNET_PILOT_LIVE=true`
- state path: `/app/state/engine-state-mainnet-pilot.json`
- journal path: `/app/journal/mainnet-pilot`

Current live-pilot risk defaults for the 99.7 USDC wallet:

- `TARGET_SCALE=5.0` (scales v16a target weights so small-account orders clear Hyperliquid minimum notional)
- `TARGET_GROSS_CAP=1.00`
- `MIN_EQUITY=50` (pilot allows slightly below 100 USDC after transfer fees)
- `MIN_AVAILABLE_BALANCE=50` (requires Hyperliquid account funds to be available for trading; unified spot USDC can count before the first perp trade)
- `MAX_EQUITY=200`
- `MAX_ORDER_NOTIONAL=50`
- `HL_LEVERAGE=5`
- `LIVE_SYMBOLS=BTC,ETH,SOL,BNB,DOGE,AVAX,ADA,SUI`

## Read-only preflight

Run only after mainnet secrets are available in environment:

```bash
HL_NETWORK=mainnet \
DRY_RUN=false \
STRATEGY_PROFILE=v16a-mainnet-pilot \
ALLOW_MAINNET_PILOT_LIVE=true \
TARGET_SCALE=5.0 \
TARGET_GROSS_CAP=1.00 \
MIN_EQUITY=50 \
MIN_AVAILABLE_BALANCE=50 \
MAX_EQUITY=200 \
MAX_ORDER_NOTIONAL=50 \
LIVE_SYMBOLS=BTC,ETH,SOL,BNB,DOGE,AVAX,ADA,SUI \
uv run python -m executor.run_mainnet_preflight
```

The command performs no exchange writes. It checks mainnet account state, open orders, symbol metadata, L2 snapshots, estimated minimum rounded order sizes, and current pilot target diagnostics.

## Deployment path

Prefer GitHub Actions CI/CD. Avoid ad hoc EC2 mutation except read-only checks and emergency stop/diagnostics.

For live pilot deployment, prefer the Deploy workflow with `target=mainnet-pilot-live`. It builds the executor image, syncs `docker-compose.prod.yml` plus `docker-compose.mainnet-pilot.yml`, and verifies the resulting container env includes `HL_NETWORK=mainnet`, `DRY_RUN=false`, `ALLOW_MAINNET_PILOT_LIVE=true`, and `STRATEGY_PROFILE=v16a-mainnet-pilot`.

Equivalent manual compose command, for emergency/operator use only:

```bash
docker compose \
  -f docker-compose.prod.yml \
  -f docker-compose.mainnet-pilot.yml \
  -f docker-compose.mainnet-pilot-live.yml \
  up -d
```

The prior dry-run phase uses `target=mainnet-pilot-dry-run` with only `docker-compose.mainnet-pilot.yml`, `DRY_RUN=true`, and `ALLOW_MAINNET_PILOT_LIVE=false`. Do not use that target when validating real order execution.

## Stop / rollback

Stop submitting new orders:

```bash
cd ~/cta-forge
docker compose -f docker-compose.prod.yml -f docker-compose.mainnet-pilot.yml stop -t 30 executor-live
```

Cancel open orders separately if needed. Flattening positions is a separate decision and should not be automatic unless there is an immediate risk reason.
