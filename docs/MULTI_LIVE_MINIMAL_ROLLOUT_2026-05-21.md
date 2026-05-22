# Multi-Live Minimal Rollout — 2026-05-21

Purpose: prepare a second real-money live instance without building a broad DBMS platform. Keep the work small, isolated, and reversible.

## Scope

Do now:

1. Isolate each live executor deployment.
2. Seed only the DB identity rows required by runtime persistence.
3. Prevent duplicate runtimes for one `LIVE_INSTANCE_ID`.
4. Use explicit env hard caps for the planned 400 USDT instance.

Do not do now:

- Full market-bar DB migration.
- DB-backed dynamic risk/config center.
- Public dashboard instance switcher.
- Secrets in DB.
- Log storage in business DB.

## Instance model

Existing instance:

- `LIVE_INSTANCE_ID=mainnet-pilot`
- container: `cta-forge-executor`
- env: `.env`
- data/state/journal: existing production paths

Planned second instance:

- `LIVE_INSTANCE_ID=mainnet-400-01`
- container: `cta-forge-executor-mainnet-400-01`
- env: `.env.mainnet-400-01`
- data/state/journal: `data-mainnet-400-01`, `state-mainnet-400-01`, `journal-mainnet-400-01`
- initial mode: dry-run only
- public dashboard status: hidden

## Runtime lock

DB-backed persistence modes now acquire a PostgreSQL advisory lock keyed by `LIVE_INSTANCE_ID` before registering the run.

- If the lock is unavailable, startup fails closed.
- The lock is held by the persistence DB connection.
- The lock releases automatically when the connection closes.
- `run_mainnet_preflight` can probe lock availability when `DATABASE_URL` and `LIVE_INSTANCE_ID` are configured.
- Set `REQUIRE_LIVE_INSTANCE_LOCK_AVAILABLE=true` only for pre-start checks; leave it unset for diagnostics while a live executor is already running.

## Minimal DB bootstrap

Use `executor.run_bootstrap_live_instance` to create only non-secret reference rows:

- `strategies`
- `strategy_profiles`
- `exchange_accounts`
- `live_instances`
- optional `public_dashboard_instances`

Example dry run:

```bash
uv run python -m executor.run_bootstrap_live_instance \
  --dry-run \
  --live-instance-id mainnet-400-01 \
  --account-id hl-mainnet-400-01 \
  --profile-slug v16a-mainnet-pilot \
  --public-instance-slug mainnet-400-01 \
  --status paused \
  --public-status hidden
```

Real write requires `DATABASE_URL` and should be done only after confirming the private env values. Keep the row `paused` while preparing secrets; switch it to `active` only for an approved dry-run/runtime start. DB-backed runtime now fails closed unless both the `live_instances` row and its `exchange_accounts` row are `active`.

## 400 USDT guard

The planned instance uses the existing `v16a-mainnet-pilot` strategy profile but has a distinct runtime gate:

- `LIVE_INSTANCE_ID=mainnet-400-01`
- `ALLOW_MAINNET_400_LIVE=true` is required before non-dry-run orders.
- `MAX_EQUITY` must be explicit and `<= 500`.
- `MAX_ORDER_NOTIONAL` must be explicit and `<= 50`, unless the existing uncapped-order override is deliberately used.
- `TARGET_GROSS_CAP <= 4`.
- `HL_LEVERAGE <= 5`.

Keep `DRY_RUN=true` until the second instance has passed isolated preflight and at least one dry-run tick.

## Rollout gates

1. Build and test code locally.
2. Copy `infra/env.mainnet-400-01.example` to a private `.env.mainnet-400-01` on the trading host.
3. Run the non-secret env guard before any runtime start:

```bash
uv run python -m executor.run_check_live_instance_env
```

4. Bootstrap DB rows with status `paused` and dashboard `hidden`.
5. After private env review, activate the DB instance/account rows for the dry-run window.
6. Run the DB-only readiness guard:

```bash
uv run python -m executor.run_check_live_instance_db --require-active --require-lock-available
```

7. Run read-only exchange/DB-lock preflight with `REQUIRE_LIVE_INSTANCE_LOCK_AVAILABLE=true`.
8. Start `executor-mainnet-400-01` under the `mainnet-400-01` compose profile with `DRY_RUN=true`.
9. Observe 1–2 hourly ticks and verify DB rows are separated by `LIVE_INSTANCE_ID`.
10. Continue a longer dry-run soak if desired; the 2026-05-22 00:00 UTC check reached `mainnet-400-01` tick 11 with DB separation intact and no real-order terms in logs.
11. Only after explicit approval: create/fund wallet, set live env flags, and promote from dry-run.

## Multi-instance DB status check

Use the read-only status CLI to summarize all live instances without ad-hoc SQL:

```bash
uv run python -m executor.run_live_instances_status \
  --require-active \
  --require-progress \
  --expect-instance mainnet-pilot \
  --expect-instance mainnet-400-01
```

It reports non-secret DB metadata for each instance, including active/paused status, public dashboard status, latest checkpoint bar, latest tick bar, latest target gross, and the latest run status. Use this before and after multi-instance changes to confirm `LIVE_INSTANCE_ID` separation.

## Pre-live wallet/account gate

Do not promote `mainnet-400-01` from dry-run while it still reuses the existing mainnet-pilot Hyperliquid address. Dry-run observation can reuse the address because it does not place orders, but real live trading must use an isolated wallet/account.

Before any funding or non-dry-run start:

1. Create or select a dedicated Hyperliquid account/address for `mainnet-400-01`.
2. Put only the approved pilot amount on that account; planned amount is 400 USDT.
3. Update the private host env only, not git:
   - `HL_PRIVATE_KEY`
   - `HL_ACCOUNT_ADDRESS`
   - notification targets if desired
4. Keep all live gates false while validating the new account:
   - `DRY_RUN=true`
   - `ALLOW_MAINNET_400_LIVE=false`
   - `ALLOW_MAINNET_PILOT_LIVE=false`
5. Bootstrap/update the non-secret DB account metadata with the dedicated address so only its hash and short prefix are stored.
6. Run env guard, DB readiness, and read-only mainnet preflight with `REQUIRE_DB_ACCOUNT_ADDRESS_MATCH=true`; this fails closed if private `HL_ACCOUNT_ADDRESS` does not match the DB account metadata.
7. Start/observe at least one more dry-run tick on the dedicated account.
8. Confirm no unmanaged positions, no open orders, and DB rows still advance only under `LIVE_INSTANCE_ID=mainnet-400-01`.
9. Only then ask for explicit approval to set:
   - `DRY_RUN=false`
   - `ALLOW_MAINNET_400_LIVE=true`

Keep `MAX_EQUITY<=500`, `MAX_ORDER_NOTIONAL<=50`, `TARGET_GROSS_CAP<=4`, and `HL_LEVERAGE<=5` unless a new review deliberately changes the 400 USDT pilot envelope.
