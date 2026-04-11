"""Funding rate carry alpha factor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class FundingRateCarryFactor:
    """Funding rate carry factor.

    Uses funding rate as a contrarian signal:
    - High positive funding → market overleveraged long → signal short
    - High negative funding → market overleveraged short → signal long

    Signal = -sign(avg_funding) * min(|z|, cap) / cap, clipped to [-1, +1].
    """

    lookback: int = 8  # number of funding periods to average (8 = 1 day for 8h funding)
    cap: float = 2.0

    @property
    def name(self) -> str:
        return f"carry_{self.lookback}"

    def compute(self, bars: pl.DataFrame) -> pl.DataFrame:
        """Compute carry signal.

        Expects columns: open_time, and optionally funding_rate.
        If funding_rate is not present, returns zero signals.
        """
        if "funding_rate" not in bars.columns:
            return pl.DataFrame(
                {
                    "open_time": bars["open_time"],
                    "signal": np.zeros(len(bars)),
                }
            )

        if len(bars) < self.lookback:
            return pl.DataFrame({"open_time": [], "signal": []}).cast(
                {"open_time": bars["open_time"].dtype, "signal": pl.Float64}
            )

        fr = bars["funding_rate"].to_numpy()
        signals = np.zeros(len(bars))

        for i in range(self.lookback, len(bars)):
            window = fr[i - self.lookback : i]
            avg = window.mean()
            std = window.std()
            if std > 1e-10:
                z = avg / std
                signals[i] = -np.sign(z) * min(abs(z), self.cap) / self.cap
            else:
                signals[i] = 0.0

        signals = np.clip(signals, -1.0, 1.0)

        return pl.DataFrame({"open_time": bars["open_time"], "signal": signals})
