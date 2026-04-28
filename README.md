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

# Run the default v10g backtest
uv run python scripts/backtest/v10g_maxrange.py

# Reproduce the current v16a research checkpoint
uv run python scripts/backtest/joint_badscore_research.py

# Run one v16a target shadow tick (dry-run only; no real orders)
DRY_RUN=true STRATEGY_PROFILE=v16a-badscore-overlay uv run python -m executor.run_shadow_tick

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

## Tech Stack

- Python 3.12, uv workspace
- FastAPI + uvicorn
- httpx (inter-service + external APIs)
- polars + numpy
- parquet (local storage)
- Docker + docker-compose
- pytest + ruff + ty
