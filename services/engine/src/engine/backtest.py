"""Backtest execution engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    final_equity: float = 0.0
    peak_equity: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0


@dataclass
class BacktestEngine:
    """Event-driven backtest engine.

    Processes bars chronologically, calling strategy functions at each step.
    """

    initial_equity: float = 10000.0
    commission_rate: float = 0.0004  # 4 bps

    def run(
        self,
        bars: dict[str, pl.DataFrame],
        compute_signals: callable,
        allocate: callable,
        check_risk: callable | None = None,
    ) -> BacktestResult:
        """Run backtest over historical bars.

        Args:
            bars: {symbol: bars_df} with aligned timestamps.
            compute_signals: fn(symbol, bars_slice) -> float signal
            allocate: fn(signals, equity) -> {symbol: target_notional}
            check_risk: optional fn(positions, bars_slice) -> adjusted positions

        Returns:
            BacktestResult with equity curve and trade log.
        """
        if not bars:
            return BacktestResult()

        # Get aligned timestamps from the first symbol
        first_symbol = next(iter(bars))
        timestamps = bars[first_symbol]["open_time"].to_list()

        equity = self.initial_equity
        peak = equity
        positions: dict[str, float] = {}  # {symbol: notional}
        entry_prices: dict[str, float] = {}

        result = BacktestResult()
        result.equity_curve.append((timestamps[0], equity))

        for t_idx in range(1, len(timestamps)):
            # Get bars up to current time for each symbol
            signals: dict[str, float] = {}
            current_bars: dict[str, pl.DataFrame] = {}

            for symbol, symbol_bars in bars.items():
                if t_idx >= len(symbol_bars):
                    continue
                bars_slice = symbol_bars[: t_idx + 1]
                current_bars[symbol] = bars_slice

                signal = compute_signals(symbol, bars_slice)
                signals[symbol] = signal

            # Get current prices
            prices = {}
            for symbol, symbol_bars in bars.items():
                if t_idx < len(symbol_bars):
                    prices[symbol] = float(symbol_bars["close"][t_idx])

            # Mark-to-market existing positions
            unrealized_pnl = 0.0
            for symbol, notional in positions.items():
                if symbol in prices and symbol in entry_prices:
                    price_change = (prices[symbol] - entry_prices[symbol]) / entry_prices[symbol]
                    unrealized_pnl += notional * price_change

            equity = self.initial_equity + unrealized_pnl
            # Account for realized P&L from closed trades
            for trade in result.trades:
                equity += trade.get("pnl", 0.0)

            # Allocate
            target = allocate(signals, equity)

            # Risk check
            if check_risk is not None:
                target = check_risk(target, current_bars, entry_prices)

            # Execute rebalance
            for symbol, target_notional in target.items():
                current = positions.get(symbol, 0.0)
                diff = target_notional - current

                if abs(diff) > 1e-6 and symbol in prices:
                    commission = abs(diff) * self.commission_rate
                    equity -= commission

                    if symbol in positions and abs(current) > 1e-6 and symbol in entry_prices:
                        # Closing/reducing position → realize P&L
                        price_change = (prices[symbol] - entry_prices[symbol]) / entry_prices[symbol]
                        pnl = current * price_change - commission
                        result.trades.append(
                            {
                                "timestamp": timestamps[t_idx],
                                "symbol": symbol,
                                "side": "close",
                                "notional": abs(current),
                                "price": prices[symbol],
                                "pnl": pnl,
                            }
                        )

                    positions[symbol] = target_notional
                    if abs(target_notional) > 1e-6:
                        entry_prices[symbol] = prices[symbol]
                    elif symbol in entry_prices:
                        del entry_prices[symbol]

            # Remove zero positions
            positions = {s: n for s, n in positions.items() if abs(n) > 1e-6}

            # Track equity
            peak = max(peak, equity)
            result.equity_curve.append((timestamps[t_idx], equity))

        result.final_equity = equity
        result.peak_equity = peak
        result.total_return = (equity - self.initial_equity) / self.initial_equity
        result.max_drawdown = max((p - e) / p if p > 0 else 0.0 for _, e in result.equity_curve for p in [peak])

        # Recalculate max drawdown properly
        running_peak = 0.0
        max_dd = 0.0
        for _, eq in result.equity_curve:
            running_peak = max(running_peak, eq)
            if running_peak > 0:
                dd = (running_peak - eq) / running_peak
                max_dd = max(max_dd, dd)
        result.max_drawdown = max_dd

        logger.info(
            "Backtest complete: return=%.2f%%, max_dd=%.2f%%, trades=%d",
            result.total_return * 100,
            result.max_drawdown * 100,
            len(result.trades),
        )

        return result
