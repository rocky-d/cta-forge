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

# Run backtest script
uv run python scripts/backtest/v10g_maxrange.py

# Full stack
docker compose up
```

## Tech Stack

- Python 3.12, uv workspace
- FastAPI + uvicorn
- httpx (inter-service + external APIs)
- polars + numpy
- parquet (local storage)
- Docker + docker-compose
- pytest + ruff + ty
