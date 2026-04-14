"""Main trading loop orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import httpx
import polars as pl
from alpha_service.factors.breakout import DonchianBreakoutFactor
from alpha_service.factors.momentum import TSMOMFactor
from core.constants import (
    ALPHA_SERVICE_URL,
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
    factors: list[str] = field(default_factory=lambda: ["tsmom_30", "breakout_15"])
    factor_weights: dict[str, float] = field(
        default_factory=lambda: {"tsmom_30": 2.0, "breakout_15": 1.0}
    )
    initial_equity: float = 10000.0
    data_service_url: str = DATA_SERVICE_URL
    alpha_service_url: str = ALPHA_SERVICE_URL
    strategy_service_url: str = STRATEGY_SERVICE_URL
    report_service_url: str = REPORT_SERVICE_URL


class TradingLoop:
    """Orchestrates the trading pipeline: data → alpha → strategy → execution."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._running = False

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
        """Compute composite signal for a symbol (sync wrapper for backtest)."""
        # In backtest mode, compute signals locally to avoid HTTP overhead
        signals = {}
        if "tsmom_30" in self.config.factors:
            factor = TSMOMFactor(lookback=30)
            result = factor.compute(bars)
            if not result.is_empty():
                signals["tsmom_30"] = float(result["signal"][-1])
        if "breakout_15" in self.config.factors:
            factor = DonchianBreakoutFactor(period=15)
            result = factor.compute(bars)
            if not result.is_empty():
                signals["breakout_15"] = float(result["signal"][-1])

        # Compose
        total_weight = (
            sum(abs(self.config.factor_weights.get(f, 1.0)) for f in signals) or 1.0
        )
        composite = (
            sum(signals[f] * self.config.factor_weights.get(f, 1.0) for f in signals)
            / total_weight
        )

        return float(max(-1.0, min(1.0, composite)))

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
