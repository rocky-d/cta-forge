# cta-forge

Crypto CTA research and execution playground.

The repository is a Python `uv` workspace with microservices for market data,
alpha research, strategy construction, execution, and reporting. It is intended
for reproducible research first, with live trading guarded behind explicit
runtime checks and private deployment configuration.

## Workspace

- `services/data-service` — historical market data access and storage
- `services/alpha-service` — alpha/factor computation
- `services/strategy-service` — signal composition, allocation, and risk helpers
- `services/executor` — backtests, target construction, and guarded live runtime
- `services/report-service` — metrics and chart rendering
- `libs/core` — shared protocols, constants, and metrics
- `libs/exchange` — exchange adapter interfaces and Hyperliquid integration

## Development

Local validation mirrors GitHub CI:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest -q
```

## Stack

- Python 3.12 + `uv`
- FastAPI + uvicorn
- httpx
- polars + numpy
- parquet local storage
- Docker / docker compose
- pytest + ruff + ty
