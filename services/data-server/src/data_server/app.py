"""FastAPI application for data-server."""

from __future__ import annotations

import os

from fastapi import FastAPI

from .routes import router
from .store import ParquetStore

DATA_DIR = os.environ.get("DATA_DIR", "./data/binance")

app = FastAPI(title="data-server", version="0.1.0")
app.include_router(router)

# Shared store instance — injected into routes via app.state
app.state.store = ParquetStore(DATA_DIR)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
