"""FastAPI application for alpha-server."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .registry import registry
from .routes import router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Auto-discover factors on startup
    registry.auto_discover()
    yield


app = FastAPI(title="alpha-service", version="0.1.0", lifespan=lifespan)
app.include_router(router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
