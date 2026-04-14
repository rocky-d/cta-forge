"""Main trading loop orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import httpx
import polars as pl
from alpha_service.factors.v10g_composite import V10GCompositeFactor
from core.constants import (
    DATA_SERVICE_URL,
    REPORT_SERVICE_URL,
    STRATEGY_SERVICE_URL,
)
from strategy_service.allocator import allocate_positions

from .backtest import BacktestEngine

logger = logging.getLogger(__name__)


class EngineMode(StrEnum):
    BACKTEST = "backtest"
    LIVE = "live"


@dataclass
class EngineConfig:
    mode: EngineMode = EngineMode.BACKTEST
    symbols: list[str] = field(default_factory=list)
    timeframe: str = "6h"
    initial_equity: float = 10000.0
    data_service_url: str = DATA_SERVICE_URL
    strategy_service_url: str = STRATEGY_SERVICE_URL
    report_service_url: str = REPORT_SERVICE_URL


class TradingLoop:
    """Orchestrates the trading pipeline: data → alpha → strategy → execution."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._running = False
        self._factor = V10GCompositeFactor()

    async def run_backtest(self) -> dict:
        """Execute a complete backtest run."""
        async with httpx.AsyncClient(timeout=60) as client:
            # 1. Fetch data
            bars = await self._fetch_data(client)
            if not bars:
                return {"error": "No data available"}

            # 2. Setup backtest engine
            engine = BacktestEngine(initial_equity=self.config.initial_equity)

            # 3. Run with strategy callbacks
            result = engine.run(
                bars=bars,
                compute_signals=lambda sym, b: self._compute_signal_sync(sym, b),
                allocate=lambda sigs, eq: self._allocate_sync(sigs, eq),
            )

            # 4. Generate report
            report = await self._generate_report(client, result)

            return {
                "status": "completed",
                "result": {
                    "final_equity": result.final_equity,
                    "total_return": result.total_return,
                    "max_drawdown": result.max_drawdown,
                    "num_trades": len(result.trades),
                },
                "report": report,
            }

    async def _fetch_data(self, client: httpx.AsyncClient) -> dict[str, pl.DataFrame]:
        """Fetch historical bars from data-service."""
        bars = {}
        for symbol in self.config.symbols:
            resp = await client.get(
                f"{self.config.data_service_url}/bars/{symbol}",
                params={"tf": self.config.timeframe},
            )
            if resp.status_code == 200:
                data = resp.json()
                if data["bars"] > 0:
                    bars[symbol] = pl.DataFrame(data["data"])
        return bars

    def _compute_signal_sync(self, symbol: str, bars: pl.DataFrame) -> float:
        """Compute v10g composite signal for a symbol."""
        return self._factor.compute_latest(bars)

    def _allocate_sync(
        self, signals: dict[str, float], equity: float
    ) -> dict[str, float]:
        """Allocate positions (sync wrapper for backtest)."""
        return allocate_positions(signals, equity)

    async def _generate_report(self, client: httpx.AsyncClient, result) -> dict:
        """Send results to report-service for metrics."""
        try:
            curve_data = [(str(t), e) for t, e in result.equity_curve]
            resp = await client.post(
                f"{self.config.report_service_url}/report",
                json={"equity_curve": curve_data, "trades": result.trades},
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning("Failed to generate report: %s", e)
        return {}
