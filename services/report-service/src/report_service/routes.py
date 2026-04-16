"""REST API routes for report-service."""

from __future__ import annotations

import base64
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel

from .metrics import calculate_metrics
from .plot import (
    plot_backtest,
    plot_drawdown,
    plot_equity_curve,
    plot_returns_distribution,
)

router = APIRouter()


class ReportRequest(BaseModel):
    equity_curve: list[tuple[str, float]]  # [(timestamp_iso, equity), ...]
    trades: list[dict] = []
    risk_free_rate: float = 0.0
    periods_per_year: float = 365 * 4


class PlotRequest(BaseModel):
    equity_curve: list[tuple[str, float]] = []
    trades: list[dict] = []
    chart_type: str = "equity"  # equity, drawdown, distribution


class BacktestPlotRequest(BaseModel):
    """Request for full three-panel backtest chart."""

    equity_curve: list[tuple[str, float]]  # [(timestamp_iso, equity), ...]
    btc_prices: list[tuple[str, float]] = []  # [(timestamp_iso, close), ...]
    eth_prices: list[tuple[str, float]] = []
    metrics: dict | None = None
    yearly: dict[str, float] | None = None
    initial_equity: float = 10_000.0
    title_extra: str = ""
    dpi: int = 200


def _parse_curve(curve: list[tuple[str, float]]) -> list[tuple[datetime, float]]:
    return [(datetime.fromisoformat(ts), eq) for ts, eq in curve]


@router.post("/report")
async def generate_report(req: ReportRequest) -> dict:
    """Generate performance report from backtest results."""
    curve = _parse_curve(req.equity_curve)
    metrics = calculate_metrics(
        curve, req.trades, req.risk_free_rate, req.periods_per_year
    )
    return {
        "metrics": {
            "total_return": f"{metrics.total_return * 100:.2f}%",
            "annualized_return": f"{metrics.annualized_return * 100:.2f}%",
            "volatility": f"{metrics.volatility * 100:.2f}%",
            "sharpe_ratio": f"{metrics.sharpe_ratio:.2f}",
            "sortino_ratio": f"{metrics.sortino_ratio:.2f}",
            "max_drawdown": f"{metrics.max_drawdown * 100:.2f}%",
            "calmar_ratio": f"{metrics.calmar_ratio:.2f}",
            "win_rate": f"{metrics.win_rate * 100:.1f}%",
            "profit_factor": f"{metrics.profit_factor:.2f}",
            "avg_trade_pnl": f"${metrics.avg_trade_pnl:.2f}",
            "num_trades": metrics.num_trades,
        },
        "raw": {
            "total_return": metrics.total_return,
            "annualized_return": metrics.annualized_return,
            "volatility": metrics.volatility,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "max_drawdown": metrics.max_drawdown,
            "calmar_ratio": metrics.calmar_ratio,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "avg_trade_pnl": metrics.avg_trade_pnl,
            "num_trades": metrics.num_trades,
        },
    }


@router.post("/plot")
async def generate_plot(req: PlotRequest) -> Response:
    """Generate chart as PNG image."""
    curve = _parse_curve(req.equity_curve) if req.equity_curve else []

    if req.chart_type == "equity":
        img_bytes = plot_equity_curve(curve)
    elif req.chart_type == "drawdown":
        img_bytes = plot_drawdown(curve)
    elif req.chart_type == "distribution":
        img_bytes = plot_returns_distribution(req.trades)
    else:
        img_bytes = plot_equity_curve(curve)

    return Response(content=img_bytes, media_type="image/png")


@router.post("/plot/base64")
async def generate_plot_base64(req: PlotRequest) -> dict:
    """Generate chart as base64-encoded PNG."""
    curve = _parse_curve(req.equity_curve) if req.equity_curve else []

    if req.chart_type == "equity":
        img_bytes = plot_equity_curve(curve)
    elif req.chart_type == "drawdown":
        img_bytes = plot_drawdown(curve)
    elif req.chart_type == "distribution":
        img_bytes = plot_returns_distribution(req.trades)
    else:
        img_bytes = plot_equity_curve(curve)

    encoded = base64.b64encode(img_bytes).decode("utf-8")
    return {"image": encoded, "media_type": "image/png"}


@router.post("/plot/backtest")
async def generate_backtest_plot(req: BacktestPlotRequest) -> Response:
    """Generate full three-panel backtest chart as PNG image."""
    curve = _parse_curve(req.equity_curve)
    btc = _parse_curve(req.btc_prices) if req.btc_prices else None
    eth = _parse_curve(req.eth_prices) if req.eth_prices else None

    img_bytes = plot_backtest(
        equity_curve=curve,
        btc_prices=btc,
        eth_prices=eth,
        metrics=req.metrics,
        yearly=req.yearly,
        title_extra=req.title_extra,
        initial_equity=req.initial_equity,
        dpi=req.dpi,
    )
    return Response(content=img_bytes, media_type="image/png")


@router.post("/plot/backtest/base64")
async def generate_backtest_plot_base64(req: BacktestPlotRequest) -> dict:
    """Generate full three-panel backtest chart as base64 PNG."""
    curve = _parse_curve(req.equity_curve)
    btc = _parse_curve(req.btc_prices) if req.btc_prices else None
    eth = _parse_curve(req.eth_prices) if req.eth_prices else None

    img_bytes = plot_backtest(
        equity_curve=curve,
        btc_prices=btc,
        eth_prices=eth,
        metrics=req.metrics,
        yearly=req.yearly,
        title_extra=req.title_extra,
        initial_equity=req.initial_equity,
        dpi=req.dpi,
    )
    encoded = base64.b64encode(img_bytes).decode("utf-8")
    return {"image": encoded, "media_type": "image/png"}
