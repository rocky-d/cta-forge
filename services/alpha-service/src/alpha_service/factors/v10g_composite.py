"""V10G composite signal factor.

The v10g strategy computes signals using an ensemble approach:
- ADX ensemble filter (multiple thresholds)
- Multi-lookback momentum voting (adaptive weights based on realized vol)
- Donchian channel direction filter (penalty for counter-trend)
- Volume ratio filter
- DI+/DI- directional filter
- Optional BTC regime filter
- Signal persistence filter

This is the canonical implementation. Both live trading and backtesting
should use this factor to ensure signal consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl


def _compute_adx(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ADX, DI+, DI- using Wilder's smoothing."""
    n = len(close)
    adx = np.zeros(n)
    dip = np.zeros(n)
    dim = np.zeros(n)

    if n < period * 2 + 1:
        return adx, dip, dim

    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if up > down and up > 0 else 0
        minus_dm[i] = down if down > up and down > 0 else 0

    atr_sum = tr[1 : period + 1].sum()
    sp = plus_dm[1 : period + 1].sum()
    sm = minus_dm[1 : period + 1].sum()

    for i in range(period + 1, n):
        atr_sum = atr_sum - atr_sum / period + tr[i]
        sp = sp - sp / period + plus_dm[i]
        sm = sm - sm / period + minus_dm[i]

        if atr_sum > 0:
            dip[i] = 100 * sp / atr_sum
            dim[i] = 100 * sm / atr_sum

        di_sum = dip[i] + dim[i]
        dx = 100 * abs(dip[i] - dim[i]) / di_sum if di_sum > 0 else 0

        if i == period * 2:
            # Seed ADX with mean of DX values
            dx_vals = []
            _atr = tr[1 : period + 1].sum()
            _sp = plus_dm[1 : period + 1].sum()
            _sm = minus_dm[1 : period + 1].sum()
            for j in range(period + 1, period * 2 + 1):
                _atr = _atr - _atr / period + tr[j]
                _sp = _sp - _sp / period + plus_dm[j]
                _sm = _sm - _sm / period + minus_dm[j]
                _dip = 100 * _sp / _atr if _atr > 0 else 0
                _dim = 100 * _sm / _atr if _atr > 0 else 0
                _ds = _dip + _dim
                dx_vals.append(100 * abs(_dip - _dim) / _ds if _ds > 0 else 0)
            adx[i] = np.mean(dx_vals) if dx_vals else 0
        elif i > period * 2:
            adx[i] = (adx[i - 1] * (period - 1) + dx) / period

    return adx, dip, dim


def _compute_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """Compute ATR using Wilder's smoothing."""
    n = len(close)
    atr = np.zeros(n)
    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
        atr[i] = (atr[i - 1] * (period - 1) + tr) / period if i >= period else tr
    return atr


@dataclass
class V10GCompositeParams:
    """Parameters for the v10g composite signal."""

    timeframe_hours: int = 6  # New: for dynamic lookbacks
    mom_lookbacks: list[int] = field(default_factory=lambda: [20, 60, 120])
    adx_threshold: float = 25.0
    adx_ensemble: list[int] = field(default_factory=lambda: [22, 27, 32])
    signal_persistence: int = 2
    donchian_period: int = 20
    rvol_lookback: int = 20
    rvol_median_lookback: int = 120
    vol_filter_lookback: int = 20
    btc_filter_lookback: int = 60


@dataclass
class V10GCompositeFactor:
    """V10G composite signal: ensemble ADX + multi-lookback momentum + filters.

    This is the single source of truth for v10g signal computation.
    Used by both the live engine and backtest scripts.
    """

    params: V10GCompositeParams = field(default_factory=V10GCompositeParams)

    @property
    def name(self) -> str:
        return "v10g_composite"

    def precompute(self, bars: pl.DataFrame) -> dict[str, np.ndarray]:
        """Precompute indicators from a bars DataFrame.

        Expects columns: open_time, open, high, low, close, volume.
        Returns dict of numpy arrays for use in compute().
        """
        c = bars["close"].to_numpy()
        h = bars["high"].to_numpy()
        lo = bars["low"].to_numpy()
        vol = bars["volume"].to_numpy()

        atr = _compute_atr(h, lo, c)
        adx, dip, dim = _compute_adx(h, lo, c)

        rets = np.zeros(len(c))
        for i in range(1, len(c)):
            rets[i] = (c[i] - c[i - 1]) / c[i - 1] if c[i - 1] > 0 else 0

        rvol = np.zeros(len(c))
        lb = self.params.rvol_lookback
        for i in range(lb, len(c)):
            rvol[i] = np.std(rets[i - lb : i])

        return {
            "close": c,
            "high": h,
            "low": lo,
            "volume": vol,
            "atr": atr,
            "adx": adx,
            "dip": dip,
            "dim": dim,
            "rvol": rvol,
        }

    def compute_signal_array(
        self,
        indicators: dict[str, np.ndarray],
        btc_indicators: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Compute raw signal array for a single symbol.

        Args:
            indicators: precomputed indicator arrays for this symbol.
            btc_indicators: precomputed indicators for BTCUSDT (optional BTC filter).

        Returns:
            1D signal array (same length as input), with persistence filter applied.
        """
        p = self.params
        c = indicators["close"]
        h = indicators["high"]
        lo = indicators["low"]
        vol = indicators["volume"]
        adx = indicators["adx"]
        dip_arr = indicators["dip"]
        dim_arr = indicators["dim"]
        rvol = indicators["rvol"]
        n = len(c)

        warmup = max(p.mom_lookbacks) + 1
        raw_sig = np.zeros(n)

        for i in range(warmup, n):
            raw_signals = []

            for adx_t in p.adx_ensemble:
                if adx[i] < adx_t:
                    raw_signals.append(0.0)
                    continue

                di_long = dip_arr[i] > dim_arr[i]
                di_short = dim_arr[i] > dip_arr[i]

                # Adaptive lookback weighting based on realized vol
                rml = p.rvol_median_lookback
                median_rvol = (
                    np.median(rvol[max(0, i - rml) : i]) if i > rml else rvol[i]
                )
                vol_ratio = rvol[i] / median_rvol if median_rvol > 0 else 1.0

                votes = 0.0
                tw = 0.0
                for j, lb in enumerate(p.mom_lookbacks):
                    if i < lb:
                        continue
                    base_w = 1.0 / (j + 1)
                    if j == 0:
                        w = base_w * min(vol_ratio, 2.0)
                    elif j == len(p.mom_lookbacks) - 1:
                        w = base_w * min(1.0 / max(vol_ratio, 0.5), 2.0)
                    else:
                        w = base_w
                    ret = (c[i] - c[i - lb]) / c[i - lb] if c[i - lb] > 0 else 0
                    votes += np.sign(ret) * w * min(abs(ret) * 20, 1.0)
                    tw += w

                raw = votes / tw if tw > 0 else 0

                # Donchian direction filter
                dp = p.donchian_period
                if i >= dp:
                    dh = h[i - dp : i].max()
                    dl = lo[i - dp : i].min()
                    dm = (dh + dl) / 2
                    dr = dh - dl
                    if dr > 1e-10:
                        donchian_pos = (c[i] - dm) / (dr / 2)
                        if raw > 0 and donchian_pos < 0:
                            raw *= 0.2
                        elif raw < 0 and donchian_pos > 0:
                            raw *= 0.2

                # Volume filter
                vfl = p.vol_filter_lookback
                if i >= vfl:
                    avg_vol = vol[i - vfl : i].mean()
                    if avg_vol > 0:
                        vr = vol[i] / avg_vol
                        if vr < 0.8:
                            raw *= 0.5
                        elif vr > 1.5:
                            raw *= 1.2

                # DI direction filter
                if raw > 0 and not di_long:
                    raw *= 0.3
                elif raw < 0 and not di_short:
                    raw *= 0.3

                # BTC regime filter
                bfl = p.btc_filter_lookback
                if btc_indicators is not None and i >= bfl:
                    bc = btc_indicators["close"]
                    if i < len(bc) and i >= bfl:
                        br = (
                            (bc[i] - bc[i - bfl]) / bc[i - bfl]
                            if bc[i - bfl] > 0
                            else 0
                        )
                        if raw > 0 and br < -0.05:
                            raw *= 0.5
                        elif raw < 0 and br > 0.05:
                            raw *= 0.5

                raw_signals.append(np.clip(raw, -1, 1))

            raw_sig[i] = np.mean(raw_signals) if raw_signals else 0

        # Signal persistence filter
        if p.signal_persistence > 1:
            filtered = np.zeros(n)
            streak = 0
            last_dir = 0.0
            for i in range(n):
                d_now = np.sign(raw_sig[i])
                if d_now == last_dir and d_now != 0:
                    streak += 1
                elif d_now != 0:
                    streak = 1
                    last_dir = d_now
                else:
                    streak = 0
                    last_dir = 0.0
                filtered[i] = raw_sig[i] if streak >= p.signal_persistence else 0
            return filtered

        return raw_sig

    def compute_latest(
        self,
        bars: pl.DataFrame,
        btc_bars: pl.DataFrame | None = None,
    ) -> float:
        """Compute the latest signal value for a single symbol.

        Convenience method for live trading.
        """
        indicators = self.precompute(bars)
        btc_ind = self.precompute(btc_bars) if btc_bars is not None else None
        signals = self.compute_signal_array(indicators, btc_ind)
        return float(signals[-1]) if len(signals) > 0 else 0.0

    def compute(self, bars: pl.DataFrame) -> pl.DataFrame:
        """Compute signal series (AlphaFactor protocol compatible).

        Returns DataFrame with open_time and signal columns.
        """
        indicators = self.precompute(bars)
        signals = self.compute_signal_array(indicators)
        warmup = max(self.params.mom_lookbacks) + 1
        return pl.DataFrame(
            {
                "open_time": bars["open_time"][warmup:],
                "signal": signals[warmup:],
            }
        )
