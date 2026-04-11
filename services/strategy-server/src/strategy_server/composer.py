"""Multi-factor signal composer."""

from __future__ import annotations

import numpy as np


def compose_signals(
    signals: dict[str, dict[str, float]],
    weights: dict[str, float],
) -> dict[str, float]:
    """Compose multiple factor signals into a single signal per symbol.

    Args:
        signals: {symbol: {factor_name: signal_value}}
        weights: {factor_name: weight}

    Returns:
        {symbol: composite_signal} clipped to [-1, +1].
    """
    result = {}
    total_weight = sum(abs(w) for w in weights.values()) or 1.0

    for symbol, factor_signals in signals.items():
        weighted_sum = 0.0
        for factor_name, weight in weights.items():
            sig = factor_signals.get(factor_name, 0.0)
            weighted_sum += sig * weight
        composite = np.clip(weighted_sum / total_weight, -1.0, 1.0)
        result[symbol] = float(composite)

    return result
