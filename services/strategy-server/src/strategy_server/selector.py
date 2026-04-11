"""Adaptive asset selection based on rolling Sharpe ratio."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import polars as pl


def select_assets(
    universe: dict[str, pl.DataFrame],
    top_n: int = 30,
    lookback: int = 90,
    min_volume: float = 0.0,
) -> list[str]:
    """Select top N assets by rolling Sharpe ratio.

    Args:
        universe: {symbol: bars_df} with columns [open_time, close, volume].
        top_n: number of assets to select.
        lookback: number of bars for Sharpe calculation.
        min_volume: minimum average daily volume filter.

    Returns:
        List of selected symbols sorted by Sharpe descending.
    """
    scores: list[tuple[str, float]] = []

    for symbol, bars in universe.items():
        if len(bars) < lookback:
            continue

        recent = bars.tail(lookback)
        close = recent["close"].to_numpy()

        # Average volume filter
        if min_volume > 0 and "volume" in recent.columns:
            avg_vol = recent["volume"].mean()
            if avg_vol is not None and float(avg_vol) < min_volume:
                continue

        # Calculate Sharpe on log returns
        log_ret = np.diff(np.log(close))
        if len(log_ret) == 0 or log_ret.std() < 1e-10:
            continue

        sharpe = log_ret.mean() / log_ret.std() * np.sqrt(len(log_ret))
        scores.append((symbol, float(sharpe)))

    # Sort by absolute Sharpe (we want assets with strong trends, long or short)
    scores.sort(key=lambda x: abs(x[1]), reverse=True)
    return [s[0] for s in scores[:top_n]]
