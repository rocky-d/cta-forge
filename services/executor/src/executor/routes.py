"""REST API routes for executor."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from .loop import EngineConfig, EngineMode, TradingLoop

router = APIRouter()

# Global state
_engine_state: dict[str, Any] = {
    "status": "idle",
    "last_result": None,
}


class BacktestRequest(BaseModel):
    symbols: list[str]
    timeframe: str = "6h"
    factors: list[str] = ["tsmom_30", "breakout_15"]
    factor_weights: dict[str, float] = {"tsmom_30": 2.0, "breakout_15": 1.0}
    initial_equity: float = 10000.0


@router.get("/status")
async def get_status() -> dict:
    return _engine_state


@router.post("/backtest")
async def run_backtest(req: BacktestRequest, background_tasks: BackgroundTasks) -> dict:
    """Start a backtest run."""
    if _engine_state["status"] == "running":
        return {"error": "Engine is already running"}

    config = EngineConfig(
        mode=EngineMode.BACKTEST,
        symbols=req.symbols,
        timeframe=req.timeframe,
        factors=req.factors,
        factor_weights=req.factor_weights,
        initial_equity=req.initial_equity,
    )

    async def _run():
        _engine_state["status"] = "running"
        try:
            loop = TradingLoop(config)
            result = await loop.run_backtest()
            _engine_state["last_result"] = result
        finally:
            _engine_state["status"] = "idle"

    background_tasks.add_task(_run)
    return {"status": "started", "symbols": req.symbols}


@router.post("/backtest/sync")
async def run_backtest_sync(req: BacktestRequest) -> dict:
    """Run backtest synchronously and return result."""
    config = EngineConfig(
        mode=EngineMode.BACKTEST,
        symbols=req.symbols,
        timeframe=req.timeframe,
        factors=req.factors,
        factor_weights=req.factor_weights,
        initial_equity=req.initial_equity,
    )

    loop = TradingLoop(config)
    result = await loop.run_backtest()
    _engine_state["last_result"] = result
    return result


@router.post("/stop")
async def stop() -> dict:
    """Stop the engine (placeholder for live mode)."""
    _engine_state["status"] = "idle"
    return {"status": "stopped"}
