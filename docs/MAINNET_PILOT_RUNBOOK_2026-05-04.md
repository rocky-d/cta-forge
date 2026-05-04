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

Initial risk defaults:

- `TARGET_GROSS_CAP=0.20`
- `MIN_EQUITY=50` (pilot allows slightly below 100 USDC after transfer fees)
- `MIN_AVAILABLE_BALANCE=50` (requires funds to be usable as perp collateral, not only spot balance)
- `MAX_EQUITY=200`
- `MAX_ORDER_NOTIONAL=25`
- `HL_LEVERAGE=5` with low real gross exposure
- `LIVE_SYMBOLS=BTC,ETH,SOL,BNB,DOGE,AVAX,ADA,SUI`

## Read-only preflight

Run only after mainnet secrets are available in environment:

```bash
HL_NETWORK=mainnet \
DRY_RUN=true \
STRATEGY_PROFILE=v16a-mainnet-pilot \
TARGET_GROSS_CAP=0.20 \
MIN_EQUITY=50 \
MIN_AVAILABLE_BALANCE=50 \
MAX_EQUITY=200 \
MAX_ORDER_NOTIONAL=25 \
LIVE_SYMBOLS=BTC,ETH,SOL,BNB,DOGE,AVAX,ADA,SUI \
uv run python -m executor.run_mainnet_preflight
```

The command performs no exchange writes. It checks mainnet account state, open orders, symbol metadata, L2 snapshots, estimated minimum rounded order sizes, and current pilot target diagnostics.

## Deployment path

Prefer GitHub Actions CI/CD. Avoid ad hoc EC2 mutation except read-only checks and emergency stop/diagnostics.

For dry-run pilot deployment, compose overlay:

```bash
docker compose -f docker-compose.prod.yml -f docker-compose.mainnet-pilot.yml up -d
```

Before live orders, explicitly change only:

- `DRY_RUN=false`
- `ALLOW_MAINNET_PILOT_LIVE=true`

## Stop / rollback

Stop submitting new orders:

```bash
cd ~/cta-forge
docker compose -f docker-compose.prod.yml -f docker-compose.mainnet-pilot.yml stop -t 30 executor-live
```

Cancel open orders separately if needed. Flattening positions is a separate decision and should not be automatic unless there is an immediate risk reason.
