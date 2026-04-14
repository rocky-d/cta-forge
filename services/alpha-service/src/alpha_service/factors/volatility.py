"""Volatility regime alpha factor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class VolatilityRegimeFactor:
    """Volatility regime factor.

    Compares short-term volatility to long-term volatility.
    - Low vol regime (short < long): trend-following environment → amplify momentum
    - High vol regime (short > long): mean-reversion environment → dampen signals

    Signal represents the regime: [-1, +1] where
    +1 = low vol (favorable for trend) and -1 = high vol (unfavorable).
    Intended as a meta-signal to weight other factors.
    """

    short_window: int = 10  # bars
    long_window: int = 50  # bars

    @property
    def name(self) -> str:
        return f"vol_regime_{self.short_window}_{self.long_window}"

    def compute(self, bars: pl.DataFrame) -> pl.DataFrame:
        """Compute volatility regime signal. Expects: open_time, close."""
        if len(bars) < self.long_window + 1:
            return pl.DataFrame({"open_time": [], "signal": []}).cast(
                {"open_time": bars["open_time"].dtype, "signal": pl.Float64}
            )

        close = bars["close"].to_numpy()
        log_ret = np.diff(np.log(close))

        signals = np.full(len(bars), np.nan)

        for i in range(self.long_window, len(log_ret) + 1):
            short_vol = log_ret[i - self.short_window : i].std()
            long_vol = log_ret[i - self.long_window : i].std()

            if long_vol > 1e-10:
                ratio = short_vol / long_vol
                # ratio < 1 → low vol → signal positive
                # ratio > 1 → high vol → signal negative
                # Map: ratio 0.5 → +1, ratio 1.0 → 0, ratio 1.5 → -1
                signal = np.clip(2.0 * (1.0 - ratio), -1.0, 1.0)
                signals[i] = signal
            else:
                signals[i] = 0.0

        # Trim NaN warm-up
        valid_mask = ~np.isnan(signals)
        return pl.DataFrame(
            {
                "open_time": bars["open_time"].filter(pl.Series(valid_mask)),
                "signal": signals[valid_mask],
            }
        )
