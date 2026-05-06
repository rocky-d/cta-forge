"""REST API routes for alpha-service."""

from __future__ import annotations

import polars as pl
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .factors.v10g_composite import V10GCompositeFactor
from .registry import registry

router = APIRouter()


class ComputeRequest(BaseModel):
    """Request body for factor computation."""

    symbol: str
    bars: list[dict]
    btc_bars: list[dict] | None = None
    factors: list[str] | None = None  # None = all registered factors


class BatchComputeRequest(BaseModel):
    """Request body for batch factor computation."""

    symbols: dict[str, list[dict]]  # {symbol: bars}
    factors: list[str] | None = None


def _compute_factor_signal(
    factor: object,
    bars_df: pl.DataFrame,
    *,
    btc_bars_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compute a factor while preserving v10g BTC regime-filter parity."""
    if isinstance(factor, V10GCompositeFactor):
        indicators = factor.precompute(bars_df)
        btc_indicators = (
            factor.precompute(btc_bars_df)
            if btc_bars_df is not None and not btc_bars_df.is_empty()
            else None
        )
        signals = factor.compute_signal_array(indicators, btc_indicators)
        warmup = max(factor.params.mom_lookbacks) + 1
        return pl.DataFrame(
            {
                "open_time": bars_df["open_time"][warmup:],
                "signal": signals[warmup:],
            }
        )
    return factor.compute(bars_df)


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
    dc_fields = getattr(factor, "__dataclass_fields__", None)
    if dc_fields is not None:
        for field_name in dc_fields:
            config[field_name] = getattr(factor, field_name)
    return {"name": factor.name, "config": config}


@router.post("/compute")
async def compute(req: ComputeRequest) -> dict:
    """Compute factor signals for a single symbol."""
    bars_df = pl.DataFrame(req.bars)
    btc_bars_df = pl.DataFrame(req.btc_bars) if req.btc_bars is not None else None

    factor_names = req.factors or registry.list_factors()
    results = {}

    for fname in factor_names:
        factor = registry.get(fname)
        if factor is None:
            raise HTTPException(status_code=404, detail=f"Factor '{fname}' not found")
        signal_df = _compute_factor_signal(
            factor,
            bars_df,
            btc_bars_df=btc_bars_df if req.symbol != "BTCUSDT" else None,
        )
        # Serialize: convert datetime to string for JSON
        if not signal_df.is_empty() and "open_time" in signal_df.columns:
            signal_df = signal_df.with_columns(
                pl.col("open_time").cast(pl.String).alias("open_time")
            )
        results[fname] = signal_df.to_dicts()

    return {"symbol": req.symbol, "signals": results}


@router.post("/compute/batch")
async def compute_batch(req: BatchComputeRequest) -> dict:
    """Compute factor signals for multiple symbols."""
    factor_names = req.factors or registry.list_factors()
    results = {}
    btc_bars = req.symbols.get("BTCUSDT")
    btc_bars_df = pl.DataFrame(btc_bars) if btc_bars is not None else None

    for symbol, bars in req.symbols.items():
        bars_df = pl.DataFrame(bars)
        symbol_signals = {}

        for fname in factor_names:
            factor = registry.get(fname)
            if factor is None:
                raise HTTPException(
                    status_code=404, detail=f"Factor '{fname}' not found"
                )
            signal_df = _compute_factor_signal(
                factor,
                bars_df,
                btc_bars_df=btc_bars_df if symbol != "BTCUSDT" else None,
            )
            if not signal_df.is_empty() and "open_time" in signal_df.columns:
                signal_df = signal_df.with_columns(
                    pl.col("open_time").cast(pl.String).alias("open_time")
                )
            symbol_signals[fname] = signal_df.to_dicts()

        results[symbol] = symbol_signals

    return {"signals": results}
