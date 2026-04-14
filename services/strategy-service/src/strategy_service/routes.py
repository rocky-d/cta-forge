"""REST API routes for strategy-service."""

from __future__ import annotations

import polars as pl
from fastapi import APIRouter
from pydantic import BaseModel

from core.constants import (
    DEFAULT_LONG_RATIO,
    DEFAULT_MAX_DRAWDOWN,
    DEFAULT_SHORT_RATIO,
    DEFAULT_TRAILING_STOP_ATR_MULT,
)

from .allocator import allocate_positions
from .composer import compose_signals
from .risk import apply_trailing_stops, check_drawdown
from .selector import select_assets

router = APIRouter()

# ── Signal routes ────────────────────────────────────────────────

signal_router = APIRouter(prefix="/signal", tags=["signal"])


class ComposeRequest(BaseModel):
    signals: dict[str, dict[str, float]]  # {symbol: {factor: signal}}
    weights: dict[str, float]  # {factor: weight}


@signal_router.post("/compose")
async def compose(req: ComposeRequest) -> dict:
    result = compose_signals(req.signals, req.weights)
    return {"composite_signals": result}


# ── Portfolio routes ─────────────────────────────────────────────

portfolio_router = APIRouter(prefix="/portfolio", tags=["portfolio"])


class SelectRequest(BaseModel):
    universe: dict[str, list[dict]]  # {symbol: bars}
    top_n: int = 30
    lookback: int = 90
    min_volume: float = 0.0


class AllocateRequest(BaseModel):
    signals: dict[str, float]  # {symbol: composite_signal}
    equity: float
    long_ratio: float = 0.7
    short_ratio: float = 0.3
    max_position_pct: float = 0.1


@portfolio_router.post("/select")
async def select(req: SelectRequest) -> dict:
    universe = {s: pl.DataFrame(bars) for s, bars in req.universe.items()}
    selected = select_assets(universe, req.top_n, req.lookback, req.min_volume)
    return {"selected": selected, "count": len(selected)}


@portfolio_router.post("/allocate")
async def allocate(req: AllocateRequest) -> dict:
    positions = allocate_positions(
        req.signals, req.equity, req.long_ratio, req.short_ratio, req.max_position_pct
    )
    return {"positions": positions}


# ── Risk routes ──────────────────────────────────────────────────

risk_router = APIRouter(prefix="/risk", tags=["risk"])


class RiskCheckRequest(BaseModel):
    positions: dict[str, float]
    bars: dict[str, list[dict]]
    entry_prices: dict[str, float]
    atr_mult: float = 2.0


class DrawdownCheckRequest(BaseModel):
    equity: float
    peak_equity: float
    max_drawdown: float = 0.15


@risk_router.post("/check")
async def check_risk(req: RiskCheckRequest) -> dict:
    bars = {s: pl.DataFrame(b) for s, b in req.bars.items()}
    adjusted = apply_trailing_stops(req.positions, bars, req.entry_prices, req.atr_mult)
    stopped = [
        s for s in req.positions if req.positions[s] != 0 and adjusted.get(s, 0) == 0
    ]
    return {"positions": adjusted, "stopped_out": stopped}


@risk_router.post("/drawdown")
async def drawdown_check(req: DrawdownCheckRequest) -> dict:
    ok = check_drawdown(req.equity, req.peak_equity, req.max_drawdown)
    dd = (
        (req.peak_equity - req.equity) / req.peak_equity if req.peak_equity > 0 else 0.0
    )
    return {"within_limits": ok, "current_drawdown": dd}


# ── Config ───────────────────────────────────────────────────────


@router.get("/config")
async def get_config() -> dict:
    return {
        "long_ratio": DEFAULT_LONG_RATIO,
        "short_ratio": DEFAULT_SHORT_RATIO,
        "max_drawdown": DEFAULT_MAX_DRAWDOWN,
        "trailing_stop_atr_mult": DEFAULT_TRAILING_STOP_ATR_MULT,
    }


# ── Register sub-routers ────────────────────────────────────────

router.include_router(signal_router)
router.include_router(portfolio_router)
router.include_router(risk_router)
