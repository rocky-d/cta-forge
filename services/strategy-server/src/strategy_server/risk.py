"""Risk management: trailing stops, drawdown control."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from cta_core.constants import DEFAULT_MAX_DRAWDOWN, DEFAULT_TRAILING_STOP_ATR_MULT

if TYPE_CHECKING:
    import polars as pl


def compute_atr(bars: pl.DataFrame, period: int = 14) -> float:
    """Compute Average True Range from recent bars."""
    if len(bars) < period + 1:
        return 0.0

    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    close = bars["close"].to_numpy()

    tr = np.zeros(len(bars) - 1)
    for i in range(1, len(bars)):
        tr[i - 1] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    return float(tr[-period:].mean())


def apply_trailing_stops(
    positions: dict[str, float],
    bars: dict[str, pl.DataFrame],
    entry_prices: dict[str, float],
    atr_mult: float = DEFAULT_TRAILING_STOP_ATR_MULT,
) -> dict[str, float]:
    """Apply ATR-based trailing stops. Zeroes out positions that hit their stop.

    Args:
        positions: {symbol: notional}
        bars: {symbol: bars_df}
        entry_prices: {symbol: entry_price}
        atr_mult: ATR multiplier for stop distance.

    Returns:
        Adjusted positions (stopped-out positions become 0).
    """
    result = dict(positions)

    for symbol, notional in positions.items():
        if abs(notional) < 1e-10:
            continue

        symbol_bars = bars.get(symbol)
        if symbol_bars is None or symbol_bars.is_empty():
            continue

        atr = compute_atr(symbol_bars)
        if atr < 1e-10:
            continue

        current_price = float(symbol_bars["close"][-1])
        entry = entry_prices.get(symbol, current_price)
        stop_distance = atr * atr_mult

        if notional > 0:
            # Long position: stop below entry
            stop_price = entry - stop_distance
            if current_price <= stop_price:
                result[symbol] = 0.0
        else:
            # Short position: stop above entry
            stop_price = entry + stop_distance
            if current_price >= stop_price:
                result[symbol] = 0.0

    return result


def check_drawdown(
    equity: float,
    peak_equity: float,
    max_drawdown: float = DEFAULT_MAX_DRAWDOWN,
) -> bool:
    """Check if portfolio drawdown exceeds threshold.

    Returns True if drawdown is within limits (ok to trade),
    False if drawdown exceeds limit (should reduce/close positions).
    """
    if peak_equity <= 0:
        return True
    dd = (peak_equity - equity) / peak_equity
    return dd < max_drawdown
