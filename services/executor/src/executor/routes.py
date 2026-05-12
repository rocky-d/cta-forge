"""REST API routes for executor."""

from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from core.metrics import calculate_live_metrics, calculate_metrics

from .backtest import BacktestResult, calc_ulcer, run_full_backtest
from .decision import V10GStrategyParams
from .journal import TradeJournal
from .live import V10G_PROFILE_SLUG, V16A_PROFILE_SLUG
from .profiles.v16a_badscore_overlay import (
    V16A_MAINNET_PILOT_PROFILE,
    validate_core_phase_hours,
)
from .run_live import (
    ALLOW_MAINNET_PILOT_UNCAPPED_ORDERS_ENV,
    MAINNET_PILOT_MAX_EQUITY,
    MAINNET_PILOT_MAX_LEVERAGE,
    MAINNET_PILOT_MAX_ORDER_NOTIONAL,
    MAINNET_PILOT_MAX_TARGET_GROSS_CAP,
)

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
        {"timestamp": t.isoformat(), "close": float(p)}
        for t, p in zip(df["open_time"].to_list(), df["close"].to_list())
        if t >= curve_start
    ]


def _format_result(result: BacktestResult) -> dict[str, Any]:
    """Format BacktestResult for JSON response."""
    if not result.equity_curve:
        return {"status": "error", "message": "No data available"}

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
            str(yr): (yearly[yr]["last"] - yearly[yr]["first"])
            / yearly[yr]["first"]
            * 100
            for yr in sorted(yearly)
        },
        "equity_curve": [
            {"timestamp": ts.isoformat(), "equity": float(eq)}
            for ts, eq in result.equity_curve
        ],
        "btc_prices": _price_series(result, "BTCUSDT"),
        "eth_prices": _price_series(result, "ETHUSDT"),
        "engine": "V10GDecisionEngine",
    }


def _is_truthy(value: str | None) -> bool:
    return (value or "").lower() in {"1", "true", "yes", "y"}


def _optional_float_from_env(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return float(value)


@router.get("/status")
async def get_status() -> dict:
    return {"status": "ready"}


@router.get("/config")
async def get_config() -> dict[str, Any]:
    """Return non-secret executor configuration for API clients and smoke tests."""
    strategy_profile = os.environ.get("STRATEGY_PROFILE", V10G_PROFILE_SLUG)
    return {
        "data_dir": os.environ.get("DATA_DIR", DATA_DIR),
        "journal_dir": os.environ.get("JOURNAL_DIR", JOURNAL_DIR),
        "state_file": os.environ.get("STATE_FILE", "engine-state.json"),
        "report_service_url": os.environ.get("REPORT_SERVICE_URL", REPORT_SERVICE_URL),
        "hl_network": os.environ.get("HL_NETWORK", "testnet"),
        "dry_run": _is_truthy(os.environ.get("DRY_RUN", "false")),
        "strategy_profile": strategy_profile,
        "default_strategy_profile": V10G_PROFILE_SLUG,
        "v16a_profile": V16A_PROFILE_SLUG,
        "v16a_mainnet_pilot_profile": V16A_MAINNET_PILOT_PROFILE.slug,
        "allow_v16a_testnet_live": _is_truthy(
            os.environ.get("ALLOW_V16A_TESTNET_LIVE")
        ),
        "allow_mainnet_pilot_live": _is_truthy(
            os.environ.get("ALLOW_MAINNET_PILOT_LIVE")
        ),
        "v16a_max_staleness_hours": float(
            os.environ.get("V16A_MAX_STALENESS_HOURS", "8")
        ),
        "v16a_core_phase_hours": validate_core_phase_hours(
            int(os.environ.get("V16A_CORE_PHASE_HOURS", "0"))
        ),
        "min_order_notional": float(os.environ.get("MIN_ORDER_NOTIONAL", "10")),
        "max_order_notional": _optional_float_from_env("MAX_ORDER_NOTIONAL"),
        "allow_mainnet_pilot_uncapped_orders": _is_truthy(
            os.environ.get(ALLOW_MAINNET_PILOT_UNCAPPED_ORDERS_ENV)
        ),
        "min_equity": _optional_float_from_env("MIN_EQUITY"),
        "min_available_balance": _optional_float_from_env("MIN_AVAILABLE_BALANCE"),
        "max_equity": _optional_float_from_env("MAX_EQUITY"),
        "target_scale": float(os.environ.get("TARGET_SCALE", "1")),
        "target_gross_cap": float(os.environ.get("TARGET_GROSS_CAP", "1")),
        "hl_leverage": int(os.environ.get("HL_LEVERAGE", "5")),
        "live_symbols": [
            symbol.strip().upper()
            for symbol in os.environ.get("LIVE_SYMBOLS", "").split(",")
            if symbol.strip()
        ],
        "mainnet_pilot_caps": {
            "max_equity": MAINNET_PILOT_MAX_EQUITY,
            "max_order_notional": MAINNET_PILOT_MAX_ORDER_NOTIONAL,
            "uncapped_orders_env": ALLOW_MAINNET_PILOT_UNCAPPED_ORDERS_ENV,
            "target_gross_cap": MAINNET_PILOT_MAX_TARGET_GROSS_CAP,
            "leverage": MAINNET_PILOT_MAX_LEVERAGE,
        },
    }


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
        "equity_records": equity_records,
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

    curve = data["equity_curve"]
    live_metrics = calculate_live_metrics(
        [(ts, eq) for ts, eq in curve],
        data["trades"],
    )
    m = live_metrics.metrics

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
            "annualized_return_raw": live_metrics.annualized_return_raw,
            "annualized_status": live_metrics.annualized_status,
            "elapsed_days": live_metrics.elapsed_days,
            "cadence_median_hours": live_metrics.cadence_median_hours,
            "record_count": live_metrics.record_count,
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
        return JSONResponse(content=data, status_code=404)

    payload = {
        "equity_records": data["equity_records"],
        "trades": data["trades"],
        "title_extra": "mainnet-pilot live | drawdown recomputed from equity records",
        "dpi": 150,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{REPORT_SERVICE_URL}/plot/live-journal",
            json=payload,
        )
        resp.raise_for_status()

    return Response(content=resp.content, media_type="image/png")
