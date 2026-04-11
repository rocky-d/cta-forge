"""Live execution engine for Hyperliquid testnet.

Runs the v10g CTA strategy on real market data:
- Every 6h (aligned to candle close): fetch bars → compute signals → rebalance
- Uses exchange-server adapter for all HL interactions
- Risk controls: max drawdown, max positions, position sizing via ATR
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from exchange_server.adapter import ExchangeAdapter

logger = logging.getLogger(__name__)

# ── v10g strategy parameters (from backtest champion) ────────────

SYMBOLS = [
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "XRP",
    "DOGE",
    "AVAX",
    "LINK",
    "ADA",
    "DOT",
    "ATOM",
    "NEAR",
]

TIMEFRAME_HOURS = 6
ADX_PERIODS = [22, 27, 32]  # ensemble ADX
ADX_THRESHOLD = 25
SIGNAL_THRESHOLD = 0.35
MIN_HOLD_BARS = 12
TRAILING_STOP_ATR = 4.5
RISK_PER_TRADE = 0.015  # 1.5%
MAX_POSITIONS = 5
REBALANCE_EVERY = 4  # bars between rebalance checks

# Risk limits
MAX_DRAWDOWN_PCT = 15.0  # hard stop: flatten everything
DD_BREAKER_PCT = 8.0  # reduce position sizes by 50%
INITIAL_EQUITY_KEY = "initial_equity"


@dataclass
class LivePosition:
    """Tracked live position with strategy metadata."""

    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    entry_bar: int
    size: Decimal
    trailing_stop: float
    highest_pnl: float = 0.0
    bars_held: int = 0


@dataclass
class LiveState:
    """Persistent state for the live trading loop."""

    positions: dict[str, LivePosition] = field(default_factory=dict)
    bar_count: int = 0
    initial_equity: float = 0.0
    peak_equity: float = 0.0
    dd_breaker_active: bool = False
    last_signals: dict[str, float] = field(default_factory=dict)


class LiveEngine:
    """Live CTA trading engine for Hyperliquid.

    Runs the v10g strategy: ensemble ADX + adaptive signals + DD breaker.
    """

    def __init__(
        self,
        exchange: ExchangeAdapter,
        *,
        symbols: list[str] | None = None,
        dry_run: bool = False,
    ) -> None:
        self._exchange = exchange
        self._symbols = symbols or SYMBOLS
        self._dry_run = dry_run
        self._running = False
        self._state = LiveState()
        self._bars_cache: dict[str, pl.DataFrame] = {}

    async def start(self) -> None:
        """Start the live trading loop."""
        logger.info("LiveEngine starting: %d symbols, dry_run=%s", len(self._symbols), self._dry_run)

        # Get initial equity
        account = await self._exchange.get_account_state()
        self._state.initial_equity = float(account.equity)
        self._state.peak_equity = self._state.initial_equity
        logger.info("Initial equity: $%.2f", self._state.initial_equity)

        self._running = True

        while self._running:
            try:
                await self._wait_for_candle_close()
                await self._tick()
            except asyncio.CancelledError:
                logger.info("LiveEngine cancelled")
                break
            except Exception:
                logger.exception("LiveEngine tick error")
                await asyncio.sleep(60)  # backoff on error

        logger.info("LiveEngine stopped")

    async def stop(self) -> None:
        """Stop the trading loop gracefully."""
        logger.info("LiveEngine stopping...")
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def state(self) -> LiveState:
        return self._state

    # ── Core loop ────────────────────────────────────────────────

    async def _wait_for_candle_close(self) -> None:
        """Sleep until next 6h candle close (UTC 00:00, 06:00, 12:00, 18:00)."""
        now = datetime.now(tz=UTC)
        hours_since_midnight = now.hour
        next_close_hour = ((hours_since_midnight // TIMEFRAME_HOURS) + 1) * TIMEFRAME_HOURS

        if next_close_hour >= 24:
            next_close = now.replace(hour=0, minute=0, second=30, microsecond=0) + timedelta(days=1)
        else:
            next_close = now.replace(hour=next_close_hour, minute=0, second=30, microsecond=0)

        wait_seconds = (next_close - now).total_seconds()
        if wait_seconds > 0:
            logger.info("Waiting %.0f seconds until next candle close (%s)", wait_seconds, next_close.strftime("%H:%M"))
            await asyncio.sleep(wait_seconds)

    async def _tick(self) -> None:
        """One iteration of the trading loop."""
        self._state.bar_count += 1
        logger.info("=== Tick #%d ===", self._state.bar_count)

        # 1. Fetch latest bars
        await self._fetch_bars()

        # 2. Check account state & risk
        account = await self._exchange.get_account_state()
        equity = float(account.equity)
        self._state.peak_equity = max(self._state.peak_equity, equity)

        drawdown_pct = (1 - equity / self._state.peak_equity) * 100 if self._state.peak_equity > 0 else 0
        logger.info("Equity: $%.2f, Peak: $%.2f, DD: %.1f%%", equity, self._state.peak_equity, drawdown_pct)

        # Hard stop
        if drawdown_pct >= MAX_DRAWDOWN_PCT:
            logger.warning("MAX DRAWDOWN BREACHED (%.1f%%) — FLATTENING ALL", drawdown_pct)
            await self._flatten_all()
            self._running = False
            return

        # DD breaker
        self._state.dd_breaker_active = drawdown_pct >= DD_BREAKER_PCT
        if self._state.dd_breaker_active:
            logger.warning("DD breaker active (%.1f%%): reducing position sizes by 50%%", drawdown_pct)

        # 3. Compute signals
        signals = self._compute_all_signals()
        self._state.last_signals = signals

        # 4. Update existing positions (trailing stops, min hold)
        await self._update_positions(equity)

        # 5. Rebalance (only every N bars)
        if self._state.bar_count % REBALANCE_EVERY == 0 or self._state.bar_count == 1:
            await self._rebalance(signals, equity)

    # ── Data fetching ────────────────────────────────────────────

    async def _fetch_bars(self) -> None:
        """Fetch latest bars from Binance for all symbols."""
        import httpx

        url = "https://fapi.binance.com/fapi/v1/klines"
        async with httpx.AsyncClient(timeout=30) as client:
            for symbol in self._symbols:
                try:
                    pair = f"{symbol}USDT"
                    resp = await client.get(
                        url,
                        params={
                            "symbol": pair,
                            "interval": "6h",
                            "limit": 200,
                        },
                    )
                    if resp.status_code != 200:
                        logger.warning("Failed to fetch %s: %d", pair, resp.status_code)
                        continue

                    raw = resp.json()
                    df = pl.DataFrame(
                        {
                            "open_time": [r[0] for r in raw],
                            "open": [float(r[1]) for r in raw],
                            "high": [float(r[2]) for r in raw],
                            "low": [float(r[3]) for r in raw],
                            "close": [float(r[4]) for r in raw],
                            "volume": [float(r[5]) for r in raw],
                        }
                    )
                    self._bars_cache[symbol] = df
                except Exception:
                    logger.exception("Error fetching %s", symbol)

    # ── Signal computation (v10g) ────────────────────────────────

    def _compute_all_signals(self) -> dict[str, float]:
        """Compute composite signals for all symbols."""
        signals = {}
        for symbol in self._symbols:
            bars = self._bars_cache.get(symbol)
            if bars is None or len(bars) < 50:
                continue
            sig = self._compute_signal(bars)
            if abs(sig) >= SIGNAL_THRESHOLD:
                signals[symbol] = sig
        logger.info("Signals: %d symbols above threshold", len(signals))
        return signals

    def _compute_signal(self, bars: pl.DataFrame) -> float:
        """Compute v10g composite signal: ensemble ADX + TSMOM + Donchian."""
        close = bars["close"].to_numpy()
        high = bars["high"].to_numpy()
        low = bars["low"].to_numpy()

        # Ensemble ADX filter
        adx_pass = False
        for period in ADX_PERIODS:
            adx = self._calc_adx(high, low, close, period)
            if adx > ADX_THRESHOLD:
                adx_pass = True
                break

        if not adx_pass:
            return 0.0

        # TSMOM signal (lookback=30)
        if len(close) > 30:
            ret = (close[-1] - close[-31]) / close[-31]
            tsmom = float(np.clip(ret * 5, -1, 1))
        else:
            tsmom = 0.0

        # Donchian breakout (period=15)
        if len(high) > 15:
            hh = float(np.max(high[-16:-1]))
            ll = float(np.min(low[-16:-1]))
            mid = (hh + ll) / 2
            rng = hh - ll if hh != ll else 1.0
            breakout = float(np.clip((close[-1] - mid) / rng * 2, -1, 1))
        else:
            breakout = 0.0

        # Weighted composite
        composite = (tsmom * 2.0 + breakout * 1.5) / 3.5
        return float(np.clip(composite, -1, 1))

    @staticmethod
    def _calc_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float:
        """Calculate ADX indicator."""
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
        dx = np.abs(plus_di[:n] - minus_di[:n]) / np.maximum(plus_di[:n] + minus_di[:n], 1e-10) * 100

        if len(dx) < period:
            return 0.0

        adx = float(np.mean(dx[-period:]))
        return adx

    # ── Position management ──────────────────────────────────────

    async def _update_positions(self, equity: float) -> None:
        """Update trailing stops and close expired positions."""
        to_close = []

        for symbol, pos in self._state.positions.items():
            pos.bars_held += 1
            bars = self._bars_cache.get(symbol)
            if bars is None:
                continue

            current_price = float(bars["close"][-1])

            # Update trailing stop
            if pos.side == "long":
                pnl_pct = (current_price - pos.entry_price) / pos.entry_price
                new_stop = current_price - self._calc_atr(bars) * TRAILING_STOP_ATR
                pos.trailing_stop = max(pos.trailing_stop, new_stop)
                if current_price <= pos.trailing_stop:
                    logger.info(
                        "Trailing stop hit: %s LONG @ %.2f (stop=%.2f)",
                        symbol,
                        current_price,
                        pos.trailing_stop,
                    )
                    to_close.append(symbol)
            else:
                pnl_pct = (pos.entry_price - current_price) / pos.entry_price
                new_stop = current_price + self._calc_atr(bars) * TRAILING_STOP_ATR
                pos.trailing_stop = min(pos.trailing_stop, new_stop)
                if current_price >= pos.trailing_stop:
                    logger.info(
                        "Trailing stop hit: %s SHORT @ %.2f (stop=%.2f)",
                        symbol,
                        current_price,
                        pos.trailing_stop,
                    )
                    to_close.append(symbol)

            pos.highest_pnl = max(pos.highest_pnl, pnl_pct)

            # Signal reversal check
            sig = self._state.last_signals.get(symbol, 0.0)
            if pos.side == "long" and sig < -SIGNAL_THRESHOLD and pos.bars_held >= MIN_HOLD_BARS:
                logger.info("Signal reversal: %s LONG → signal=%.2f", symbol, sig)
                to_close.append(symbol)
            elif pos.side == "short" and sig > SIGNAL_THRESHOLD and pos.bars_held >= MIN_HOLD_BARS:
                logger.info("Signal reversal: %s SHORT → signal=%.2f", symbol, sig)
                to_close.append(symbol)

        for symbol in set(to_close):
            await self._close_position(symbol)

    async def _rebalance(self, signals: dict[str, float], equity: float) -> None:
        """Open new positions based on signals."""
        current_count = len(self._state.positions)
        available_slots = MAX_POSITIONS - current_count
        if available_slots <= 0:
            return

        # Sort by signal strength
        candidates = [(sym, sig) for sym, sig in signals.items() if sym not in self._state.positions]
        candidates.sort(key=lambda x: abs(x[1]), reverse=True)

        for symbol, signal in candidates[:available_slots]:
            await self._open_position(symbol, signal, equity)

    async def _open_position(self, symbol: str, signal: float, equity: float) -> None:
        """Open a new position."""
        bars = self._bars_cache.get(symbol)
        if bars is None:
            return

        atr = self._calc_atr(bars)
        if atr <= 0:
            return

        current_price = float(bars["close"][-1])

        # Position sizing: risk-based (ATR)
        risk_amount = equity * RISK_PER_TRADE
        if self._state.dd_breaker_active:
            risk_amount *= 0.5

        size_usd = risk_amount / (atr * TRAILING_STOP_ATR / current_price)
        size_usd = min(size_usd, equity * 0.2)  # max 20% per position
        size = Decimal(str(size_usd / current_price))

        is_buy = signal > 0
        side = "long" if is_buy else "short"

        logger.info(
            "Opening %s %s: size=$%.0f (%.4f), signal=%.2f, ATR=%.2f",
            side.upper(),
            symbol,
            size_usd,
            float(size),
            signal,
            atr,
        )

        if self._dry_run:
            logger.info("[DRY RUN] Would place %s %s %.4f @ ~%.2f", side, symbol, float(size), current_price)
        else:
            result = await self._exchange.place_market_order(symbol, is_buy, size)
            if not result.success:
                logger.error("Failed to open %s %s: %s", side, symbol, result.message)
                return
            if result.avg_price > 0:
                current_price = result.avg_price

        # Set trailing stop
        stop = (current_price - atr * TRAILING_STOP_ATR) if is_buy else (current_price + atr * TRAILING_STOP_ATR)

        self._state.positions[symbol] = LivePosition(
            symbol=symbol,
            side=side,
            entry_price=current_price,
            entry_bar=self._state.bar_count,
            size=size,
            trailing_stop=stop,
        )
        logger.info("Position opened: %s %s @ %.2f, stop=%.2f", side.upper(), symbol, current_price, stop)

    async def _close_position(self, symbol: str) -> None:
        """Close an existing position."""
        pos = self._state.positions.get(symbol)
        if pos is None:
            return

        is_buy = pos.side == "short"  # reverse to close
        logger.info("Closing %s %s: size=%.4f, held=%d bars", pos.side.upper(), symbol, float(pos.size), pos.bars_held)

        if not self._dry_run:
            result = await self._exchange.place_market_order(symbol, is_buy, pos.size, reduce_only=True)
            if not result.success:
                logger.error("Failed to close %s: %s", symbol, result.message)
                return

        del self._state.positions[symbol]
        logger.info("Position closed: %s %s", pos.side.upper(), symbol)

    async def _flatten_all(self) -> None:
        """Emergency: close all positions."""
        logger.warning("FLATTENING ALL POSITIONS")
        symbols = list(self._state.positions.keys())
        for symbol in symbols:
            await self._close_position(symbol)
        # Also cancel any open orders
        if not self._dry_run:
            await self._exchange.cancel_all_orders()

    @staticmethod
    def _calc_atr(bars: pl.DataFrame, period: int = 14) -> float:
        """Calculate ATR from bars."""
        if len(bars) < period + 1:
            return 0.0
        high = bars["high"].to_numpy()
        low = bars["low"].to_numpy()
        close = bars["close"].to_numpy()
        tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
        tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))
        return float(np.mean(tr[-period:]))
