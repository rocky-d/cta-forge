"""Donchian channel breakout alpha factor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class DonchianBreakoutFactor:
    """Donchian channel breakout factor.

    Signal = +1 if close breaks above N-period high.
    Signal = -1 if close breaks below N-period low.
    Signal = 0 otherwise.
    Optionally filtered by ADX threshold.
    """

    period: int = 15
    adx_period: int = 14
    adx_threshold: float = 25.0  # 0 to disable ADX filter

    @property
    def name(self) -> str:
        return f"breakout_{self.period}"

    def compute(self, bars: pl.DataFrame) -> pl.DataFrame:
        """Compute breakout signal. Expects: open_time, high, low, close."""
        n = max(self.period, self.adx_period) + 1
        if len(bars) < n:
            return pl.DataFrame({"open_time": [], "signal": []}).cast(
                {"open_time": bars["open_time"].dtype, "signal": pl.Float64}
            )

        high = bars["high"].to_numpy()
        low = bars["low"].to_numpy()
        close = bars["close"].to_numpy()

        signals = np.zeros(len(bars))

        for i in range(self.period, len(bars)):
            upper = high[i - self.period : i].max()
            lower = low[i - self.period : i].min()

            if close[i] > upper:
                signals[i] = 1.0
            elif close[i] < lower:
                signals[i] = -1.0

        # ADX filter
        if self.adx_threshold > 0:
            adx = self._compute_adx(high, low, close)
            for i in range(len(signals)):
                if adx[i] < self.adx_threshold:
                    signals[i] = 0.0

        # Trim warm-up period
        valid_start = n - 1
        return pl.DataFrame(
            {
                "open_time": bars["open_time"][valid_start:],
                "signal": signals[valid_start:],
            }
        )

    def _compute_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Compute ADX indicator."""
        n = len(high)
        adx = np.zeros(n)

        if n < self.adx_period + 1:
            return adx

        # True Range
        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)

        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]
            plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
            minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0

        # Smoothed averages (Wilder's smoothing)
        p = self.adx_period
        atr = np.zeros(n)
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        dx = np.zeros(n)

        atr[p] = tr[1 : p + 1].sum()
        s_plus = plus_dm[1 : p + 1].sum()
        s_minus = minus_dm[1 : p + 1].sum()

        for i in range(p + 1, n):
            atr[i] = atr[i - 1] - atr[i - 1] / p + tr[i]
            s_plus = s_plus - s_plus / p + plus_dm[i]
            s_minus = s_minus - s_minus / p + minus_dm[i]

            if atr[i] > 0:
                plus_di[i] = 100 * s_plus / atr[i]
                minus_di[i] = 100 * s_minus / atr[i]

            di_sum = plus_di[i] + minus_di[i]
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum if di_sum > 0 else 0

        # ADX = smoothed DX
        if n > 2 * p:
            adx[2 * p] = dx[p + 1 : 2 * p + 1].mean()
            for i in range(2 * p + 1, n):
                adx[i] = (adx[i - 1] * (p - 1) + dx[i]) / p

        return adx
