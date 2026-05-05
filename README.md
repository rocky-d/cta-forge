# cta-forge

Crypto CTA strategy forge, monorepo microservices architecture.

## Services

| Service | Description |
|---|---|
| data-service | Binance historical data provider |
| alpha-service | Alpha factor computation |
| strategy-service | Signal composition, asset selection, allocation, risk |
| executor | Backtest & live execution |
| report-service | Performance metrics & visualization |

## Libraries

| Library | Description |
|---|---|
| core | Shared protocols, constants, metrics |
| exchange | Exchange connectivity (Hyperliquid adapter) |

Ports and service URLs are configured in `core/constants.py` with sensible defaults, overridable via environment variables (see `.env.example`).

## Quick Start

```bash
# Install all workspace members
uv sync

# Run a single service (dev)
cd services/data-service && uv run uvicorn data_service.app:app --reload

# Reproduce the default v10g backtest
# scripts/backtest is a thin CLI wrapper; strategy logic lives in executor.*
uv run python scripts/backtest/v10g_maxrange.py

# Reproduce the current v16a research checkpoint and three-panel chart
# Core logic lives in executor.profiles.v16a_badscore_overlay,
# executor.signal_pipeline, and executor.portfolio_backtest.
uv run python scripts/backtest/joint_badscore_research.py

# Run one v16a target shadow tick (dry-run only; no real orders)
# Requires HL_PRIVATE_KEY/HL_ACCOUNT_ADDRESS in the environment or sourced .env.
DRY_RUN=true STRATEGY_PROFILE=v16a-badscore-overlay uv run python -m executor.run_shadow_tick

# Inspect target-mode diagnostics after a shadow/live run
cat journal/shadow-v16a/targets.jsonl
cat journal/targets.jsonl

# Full stack
docker compose up
```

## Validation

Local checks mirror the GitHub Lint/Test workflows:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest -q
```

Deployment is manual via GitHub Actions `workflow_dispatch`; do not use deploys for strategy experiments.

v16a is shadow-safe by default: use `DRY_RUN=true STRATEGY_PROFILE=v16a-badscore-overlay` for observation. Non-dry-run v16a requires either `HL_NETWORK=testnet` plus explicit `ALLOW_V16A_TESTNET_LIVE=true`, or the separate guarded `v16a-mainnet-pilot` profile plus `ALLOW_MAINNET_PILOT_LIVE=true`. Current mainnet rollout is the small-account live pilot via GitHub Actions target `mainnet-pilot-live`; see `docs/MAINNET_PILOT_RUNBOOK_2026-05-04.md` for the active caps and checks.

## Tech Stack

- Python 3.12, uv workspace
- FastAPI + uvicorn
- httpx (inter-service + external APIs)
- polars + numpy
- parquet (local storage)
- Docker + docker-compose
- pytest + ruff + ty
