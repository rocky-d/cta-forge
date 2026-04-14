"""Time-Series Momentum (TSMOM) alpha factor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class TSMOMFactor:
    """Time-Series Momentum factor.

    Computes normalized returns over a lookback period as a signal.
    Signal = sign(return) * min(|z-score|, cap) / cap
    Clipped to [-1, +1].
    """

    lookback: int = 30  # number of bars
    cap: float = 3.0  # z-score cap for normalization

    @property
    def name(self) -> str:
        return f"tsmom_{self.lookback}"

    def compute(self, bars: pl.DataFrame) -> pl.DataFrame:
        """Compute TSMOM signal from bars DataFrame.

        Expects columns: open_time, close.
        Returns DataFrame with: open_time, signal.
        """
        if len(bars) < self.lookback + 1:
            return pl.DataFrame({"open_time": [], "signal": []}).cast(
                {"open_time": bars["open_time"].dtype, "signal": pl.Float64}
            )

        close = bars["close"].to_numpy()

        # Period return: r = close[t] / close[t - lookback] - 1
        returns = close[self.lookback :] / close[: -self.lookback] - 1

        # Rolling std of log returns for normalization
        log_ret = np.diff(np.log(close))
        rolling_std = np.array(
            [
                log_ret[max(0, i - self.lookback + 1) : i + 1].std()
                for i in range(len(log_ret))
            ]
        )
        # Align: rolling_std starts at index 0 of log_ret, returns starts at index (lookback-1)
        rolling_std_aligned = rolling_std[self.lookback - 1 :]

        # Z-score
        epsilon = 1e-10
        z = returns / (rolling_std_aligned + epsilon)

        # Normalize to [-1, +1]
        signal = np.sign(z) * np.minimum(np.abs(z), self.cap) / self.cap
        signal = np.clip(signal, -1.0, 1.0)

        timestamps = bars["open_time"][self.lookback :]

        return pl.DataFrame({"open_time": timestamps, "signal": signal})
