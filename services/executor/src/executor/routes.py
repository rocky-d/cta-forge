"""REST API routes for executor."""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from .backtest import BacktestResult, calc_ulcer, run_full_backtest

router = APIRouter()

DATA_DIR = os.environ.get("DATA_DIR", "./data")


class BacktestRequest(BaseModel):
    symbols: list[str] | None = None
    timeframe: str = "6h"
    initial_equity: float = 10000.0


def _format_result(result: BacktestResult) -> dict[str, Any]:
    """Format BacktestResult for JSON response."""
    if not result.equity_curve:
        return {"status": "error", "message": "No data available"}

    from core.metrics import calculate_metrics

    m = calculate_metrics(result.equity_curve, result.trades)
    ulcer = calc_ulcer(result.equity_curve)

    # Yearly breakdown
    yearly: dict[int, dict[str, float]] = {}
    for ts, eq in result.equity_curve:
        yr = ts.year
        if yr not in yearly:
            yearly[yr] = {"first": eq}
        yearly[yr]["last"] = eq

    return {
        "status": "completed",
        "period": f"{result.start_date} -> {result.end_date}",
        "days": result.days,
        "symbols": result.symbols,
        "num_symbols": len(result.symbols),
        "initial_equity": result.initial_equity,
        "metrics": {
            "total_return": m.total_return,
            "annualized_return": m.annualized_return,
            "sharpe_ratio": m.sharpe_ratio,
            "sortino_ratio": m.sortino_ratio,
            "max_drawdown": m.max_drawdown,
            "calmar_ratio": m.calmar_ratio,
            "profit_factor": m.profit_factor,
            "win_rate": m.win_rate,
            "num_trades": m.num_trades,
            "ulcer_index": ulcer,
        },
        "yearly": {
            str(yr): round(
                (yearly[yr]["last"] - yearly[yr]["first"]) / yearly[yr]["first"] * 100,
                1,
            )
            for yr in sorted(yearly)
        },
        "equity_curve": [
            {"timestamp": ts.isoformat(), "equity": round(eq, 2)}
            for ts, eq in result.equity_curve
        ],
        "engine": "V10GDecisionEngine",
    }


@router.get("/status")
async def get_status() -> dict:
    return {"status": "ready"}


@router.post("/backtest")
async def run_backtest_endpoint(req: BacktestRequest) -> dict:
    """Run backtest synchronously and return full results."""
    result = await run_full_backtest(
        data_dir=DATA_DIR,
        symbols=req.symbols,
        timeframe=req.timeframe,
        initial_equity=req.initial_equity,
    )
    return _format_result(result)


@router.post("/backtest/sync")
async def run_backtest_sync(req: BacktestRequest) -> dict:
    """Alias for /backtest (kept for backward compatibility)."""
    return await run_backtest_endpoint(req)
