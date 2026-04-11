"""REST API routes for alpha-server."""

from __future__ import annotations

import polars as pl
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .registry import registry

router = APIRouter()


class ComputeRequest(BaseModel):
    """Request body for factor computation."""

    symbol: str
    bars: list[dict]
    factors: list[str] | None = None  # None = all registered factors


class BatchComputeRequest(BaseModel):
    """Request body for batch factor computation."""

    symbols: dict[str, list[dict]]  # {symbol: bars}
    factors: list[str] | None = None


@router.get("/factors")
async def list_factors() -> dict[str, list[str]]:
    return {"factors": registry.list_factors()}


@router.get("/factors/{name}/config")
async def factor_config(name: str) -> dict:
    factor = registry.get(name)
    if factor is None:
        raise HTTPException(status_code=404, detail=f"Factor '{name}' not found")
    # Return dataclass fields as config
    config = {}
    if hasattr(factor, "__dataclass_fields__"):
        for field_name in factor.__dataclass_fields__:
            config[field_name] = getattr(factor, field_name)
    return {"name": factor.name, "config": config}


@router.post("/compute")
async def compute(req: ComputeRequest) -> dict:
    """Compute factor signals for a single symbol."""
    bars_df = pl.DataFrame(req.bars)

    factor_names = req.factors or registry.list_factors()
    results = {}

    for fname in factor_names:
        factor = registry.get(fname)
        if factor is None:
            raise HTTPException(status_code=404, detail=f"Factor '{fname}' not found")
        signal_df = factor.compute(bars_df)
        # Serialize: convert datetime to string for JSON
        if not signal_df.is_empty() and "open_time" in signal_df.columns:
            signal_df = signal_df.with_columns(pl.col("open_time").cast(pl.String).alias("open_time"))
        results[fname] = signal_df.to_dicts()

    return {"symbol": req.symbol, "signals": results}


@router.post("/compute/batch")
async def compute_batch(req: BatchComputeRequest) -> dict:
    """Compute factor signals for multiple symbols."""
    factor_names = req.factors or registry.list_factors()
    results = {}

    for symbol, bars in req.symbols.items():
        bars_df = pl.DataFrame(bars)
        symbol_signals = {}

        for fname in factor_names:
            factor = registry.get(fname)
            if factor is None:
                raise HTTPException(status_code=404, detail=f"Factor '{fname}' not found")
            signal_df = factor.compute(bars_df)
            if not signal_df.is_empty() and "open_time" in signal_df.columns:
                signal_df = signal_df.with_columns(pl.col("open_time").cast(pl.String).alias("open_time"))
            symbol_signals[fname] = signal_df.to_dicts()

        results[symbol] = symbol_signals

    return {"signals": results}
