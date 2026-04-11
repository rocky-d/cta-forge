# cta-forge

Crypto CTA strategy forge — monorepo microservices architecture.

## Services

| Service | Port | Description |
|---|---|---|
| data-server | 8001 | Binance historical data provider |
| alpha-server | 8002 | Alpha factor computation |
| strategy-server | 8003 | Signal composition, asset selection, allocation, risk |
| engine | 8004 | Backtest & live execution |
| reporter | 8005 | Performance metrics & visualization |

## Shared Library

- `cta-core` — types, protocols, constants

## Quick Start

```bash
# Install all workspace members
uv sync

# Run a single service (dev)
cd services/data-server && uv run uvicorn data_server.app:app --reload --port 8001

# Run backtest
cd services/engine && uv run python -m engine --mode backtest

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
