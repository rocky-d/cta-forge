"""Position allocation with asymmetric long/short support."""

from __future__ import annotations

from cta_core.constants import DEFAULT_LONG_RATIO, DEFAULT_SHORT_RATIO


def allocate_positions(
    signals: dict[str, float],
    equity: float,
    long_ratio: float = DEFAULT_LONG_RATIO,
    short_ratio: float = DEFAULT_SHORT_RATIO,
    max_position_pct: float = 0.1,
) -> dict[str, float]:
    """Allocate capital to positions based on signals.

    Args:
        signals: {symbol: signal} where signal in [-1, +1].
        equity: total portfolio equity.
        long_ratio: fraction of capital for long side (default 0.7).
        short_ratio: fraction of capital for short side (default 0.3).
        max_position_pct: max single position as fraction of equity.

    Returns:
        {symbol: target_notional}. Positive = long, negative = short.
    """
    if not signals:
        return {}

    longs = {s: v for s, v in signals.items() if v > 0}
    shorts = {s: v for s, v in signals.items() if v < 0}

    result: dict[str, float] = {}

    # Allocate long side
    long_budget = equity * long_ratio
    long_total_signal = sum(longs.values()) or 1.0
    for symbol, sig in longs.items():
        raw = long_budget * (sig / long_total_signal)
        capped = min(abs(raw), equity * max_position_pct)
        result[symbol] = capped

    # Allocate short side
    short_budget = equity * short_ratio
    short_total_signal = sum(abs(v) for v in shorts.values()) or 1.0
    for symbol, sig in shorts.items():
        raw = short_budget * (abs(sig) / short_total_signal)
        capped = min(abs(raw), equity * max_position_pct)
        result[symbol] = -capped

    return result
