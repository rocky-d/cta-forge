"""Shared technical indicators used across engine modules.

Extracted from live.py to avoid duplication with loop.py and risk.py.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def calc_adx(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> float:
    """Calculate ADX (Average Directional Index).

    Uses simple moving average smoothing (not Wilder's).
    Returns the current ADX value as a float.
    """
    if len(close) < period * 2:
        return 0.0

    # True Range
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))

    # Directional Movement
    up = high[1:] - high[:-1]
    down = low[:-1] - low[1:]
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    # Smoothed averages
    atr = np.convolve(tr, np.ones(period) / period, mode="valid")
    plus_di = np.convolve(plus_dm, np.ones(period) / period, mode="valid")
    minus_di = np.convolve(minus_dm, np.ones(period) / period, mode="valid")

    # Avoid division by zero
    atr = np.maximum(atr, 1e-10)
    plus_di = (plus_di / atr[: len(plus_di)]) * 100
    minus_di = (minus_di / atr[: len(minus_di)]) * 100

    n = min(len(plus_di), len(minus_di))
    dx = (
        np.abs(plus_di[:n] - minus_di[:n])
        / np.maximum(plus_di[:n] + minus_di[:n], 1e-10)
        * 100
    )

    if len(dx) < period:
        return 0.0

    return float(np.mean(dx[-period:]))


def calc_atr(bars: pl.DataFrame, period: int = 14) -> float:
    """Calculate ATR (Average True Range) from bars DataFrame.

    Expects columns: high, low, close.
    Returns the mean true range over the last `period` bars.
    """
    if len(bars) < period + 1:
        return 0.0
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    close = bars["close"].to_numpy()
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))
    return float(np.mean(tr[-period:]))
