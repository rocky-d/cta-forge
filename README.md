# cta-forge

Crypto CTA strategy forge — monorepo microservices architecture.

## Services

| Service | Port | Description |
|---|---|---|
| data-service | 8001 | Binance historical data provider |
| alpha-service | 8002 | Alpha factor computation |
| strategy-service | 8003 | Signal composition, asset selection, allocation, risk |
| executor | 8004 | Backtest & live execution |
| report-service | 8005 | Performance metrics & visualization |

## Libraries

| Library | Description |
|---|---|
| core | Shared types, protocols, constants |
| exchange | Exchange connectivity (Hyperliquid adapter) |

## Quick Start

```bash
# Install all workspace members
uv sync

# Run a single service (dev)
cd services/data-service && uv run uvicorn data_service.app:app --reload --port 8001

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
