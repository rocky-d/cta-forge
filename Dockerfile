# Multi-stage Dockerfile for cta-forge services.
# Build from repo root:
#   docker build --target data-service -t cta-forge/data-service .
#   docker build --target executor -t cta-forge/executor .

FROM python:3.12-slim AS base

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy workspace config + lockfile first (layer cache)
COPY pyproject.toml uv.lock ./

# Copy all libs and services (uv workspace needs full structure)
COPY libs/ libs/
COPY services/ services/

# Sync all packages
RUN uv sync --frozen --no-dev --all-packages

# ── Service targets ──────────────────────────────────────────

FROM base AS data-service
EXPOSE 8001
CMD ["uv", "run", "uvicorn", "data_service.app:app", "--host", "0.0.0.0", "--port", "8001"]

FROM base AS alpha-service
EXPOSE 8002
CMD ["uv", "run", "uvicorn", "alpha_service.app:app", "--host", "0.0.0.0", "--port", "8002"]

FROM base AS strategy-service
EXPOSE 8003
CMD ["uv", "run", "uvicorn", "strategy_service.app:app", "--host", "0.0.0.0", "--port", "8003"]

FROM base AS executor
EXPOSE 8004
CMD ["uv", "run", "uvicorn", "executor.app:app", "--host", "0.0.0.0", "--port", "8004"]

FROM base AS report-service
EXPOSE 8005
CMD ["uv", "run", "uvicorn", "report_service.app:app", "--host", "0.0.0.0", "--port", "8005"]
