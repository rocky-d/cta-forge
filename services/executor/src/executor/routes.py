"""REST API routes for executor."""

from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import APIRouter
from pydantic import BaseModel

from .backtest import BacktestResult, calc_ulcer, run_full_backtest
from .decision import V10GStrategyParams
from .journal import TradeJournal

router = APIRouter()

DATA_DIR = os.environ.get("DATA_DIR", "./data")
JOURNAL_DIR = os.environ.get("JOURNAL_DIR", "journal")
REPORT_SERVICE_URL = os.environ.get("REPORT_SERVICE_URL", "http://localhost:8005")


class BacktestRequest(BaseModel):
    symbols: list[str] | None = None
    timeframe: str = "6h"
    initial_equity: float = 10000.0


def _price_series(result: BacktestResult, symbol: str) -> list[dict[str, Any]]:
    """Extract price series for a symbol from backtest result bars."""
    df = result.bars.get(symbol)
    if df is None or df.is_empty() or not result.equity_curve:
        return []
    curve_start = result.equity_curve[0][0]
    return [
        {"timestamp": t.isoformat(), "close": round(float(p), 2)}
        for t, p in zip(df["open_time"].to_list(), df["close"].to_list())
        if t >= curve_start
    ]


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
        "btc_prices": _price_series(result, "BTCUSDT"),
        "eth_prices": _price_series(result, "ETHUSDT"),
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
        initial_equity=req.initial_equity,
        params=V10GStrategyParams(timeframe_str=req.timeframe),
    )
    return _format_result(result)


@router.post("/backtest/sync")
async def run_backtest_sync(req: BacktestRequest) -> dict:
    """Alias for /backtest (kept for backward compatibility)."""
    return await run_backtest_endpoint(req)


# ---------------------------------------------------------------------------
# Live report endpoints
# ---------------------------------------------------------------------------


def _journal_to_report_format(
    journal: TradeJournal,
) -> dict[str, Any]:
    """Convert journal JSONL data into report-service compatible format."""
    equity_records = journal.load_equity()
    trade_records = journal.load_trades()

    if not equity_records:
        return {"status": "error", "message": "No equity data found"}

    # equity_curve: [(timestamp_iso, equity), ...]
    equity_curve = [(r["ts"], r["equity"]) for r in equity_records]

    # trades: only closed trades (with pnl field)
    closed_trades = [r for r in trade_records if "pnl" in r]

    # current positions from latest equity record
    latest = equity_records[-1]
    positions = latest.get("positions", {})

    return {
        "equity_curve": equity_curve,
        "trades": closed_trades,
        "positions": positions,
        "bars": latest.get("bar", 0),
        "n_positions": latest.get("n_positions", 0),
    }


@router.get("/report/live")
async def get_live_report() -> dict[str, Any]:
    """Generate live performance report with metrics."""
    journal = TradeJournal(JOURNAL_DIR)
    data = _journal_to_report_format(journal)

    if data.get("status") == "error":
        return data

    from core.metrics import calculate_metrics

    curve = data["equity_curve"]
    m = calculate_metrics(
        [(ts, eq) for ts, eq in curve],
        data["trades"],
    )

    first_eq = curve[0][1]
    last_eq = curve[-1][1]

    return {
        "status": "ok",
        "period": f"{curve[0][0]} -> {curve[-1][0]}",
        "bars": data["bars"],
        "initial_equity": first_eq,
        "current_equity": last_eq,
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
        },
        "equity_curve": [{"timestamp": ts, "equity": eq} for ts, eq in curve],
        "positions": data["positions"],
    }


@router.get("/report/live/plot")
async def get_live_report_plot() -> Any:
    """Generate live performance chart via report-service."""
    journal = TradeJournal(JOURNAL_DIR)
    data = _journal_to_report_format(journal)

    if data.get("status") == "error":
        from fastapi.responses import JSONResponse

        return JSONResponse(content=data, status_code=404)

    curve = data["equity_curve"]
    first_eq = curve[0][1]

    payload = {
        "equity_curve": curve,
        "initial_equity": first_eq,
        "title_extra": "Live",
        "dpi": 150,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{REPORT_SERVICE_URL}/plot/backtest", json=payload)
        resp.raise_for_status()

    from fastapi.responses import Response

    return Response(content=resp.content, media_type="image/png")
