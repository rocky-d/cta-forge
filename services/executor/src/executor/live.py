"""Live execution engine for Hyperliquid.

Runs the v10g CTA strategy on real market data:
- Every 6h (aligned to candle close): fetch bars → compute signals → decide → execute
- Uses V10GDecisionEngine for all trade decisions (shared with backtest)
- Uses exchange adapter for all HL interactions
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import httpx
import numpy as np
import polars as pl
from lark_bots import ABot

if TYPE_CHECKING:
    from exchange.adapter import ExchangeAdapter

from alpha_service.factors.v10g_composite import V10GCompositeFactor

from .decision import (
    ActionKind,
    BarSnapshot,
    EngineState,
    PositionState,
    TradeAction,
    V10GDecisionEngine,
    V10GStrategyParams,
)
from .journal import TradeJournal

logger = logging.getLogger(__name__)


# ── Notification protocol ────────────────────────────────────────


class _Notifier:
    """Protocol-ish base for trade notifications."""

    async def send(self, message: str) -> None:
        """Send a notification message."""


class _NullNotifier(_Notifier):
    """No-op notifier for when notifications aren't configured."""

    async def send(self, message: str) -> None:
        pass


class TelegramNotifier(_Notifier):
    """Send notifications via Telegram Bot API."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id

    async def send(self, message: str) -> None:
        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(
                    url,
                    json={
                        "chat_id": self._chat_id,
                        "text": message,
                        "parse_mode": "Markdown",
                    },
                )
        except Exception as e:
            logger.warning("Telegram notification failed: %s", e)


class LarkNotifier(_Notifier):
    """Send notifications via Lark/Feishu webhook (lark-bots)."""

    def __init__(self, webhook_url: str, secret: str | None = None) -> None:
        self._webhook_url = webhook_url
        self._secret = secret

    async def send(self, message: str) -> None:
        try:
            async with ABot(self._webhook_url, secret=self._secret) as bot:
                await bot.asend_text(message)
        except Exception as e:
            logger.warning("Lark notification failed: %s", e)


class MultiNotifier(_Notifier):
    """Fan-out notifier: sends to all backends concurrently."""

    def __init__(self, notifiers: list[_Notifier]) -> None:
        self._notifiers = notifiers

    async def send(self, message: str) -> None:
        await asyncio.gather(
            *(n.send(message) for n in self._notifiers),
            return_exceptions=True,
        )


# ── v10g strategy defaults ───────────────────────────────────────

# Symbols to trade (12 main symbols, the full 19-symbol backtest adds more)
V10G_SYMBOLS = [
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "DOGE",
    "AVAX",
    "ADA",
    "ATOM",
    "NEAR",
]

TIMEFRAME_HOURS = 6


# ── Legacy types for state persistence compatibility ─────────────
# These are re-exported for state.py which imports them.


@dataclass
class LivePosition:
    """Tracked live position with strategy metadata.

    Note: This is kept for state.py backward compat. Internally we use PositionState.
    """

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
    """Persistent state for the live trading loop.

    Note: This is kept for state.py backward compat. Internally we use EngineState.
    """

    positions: dict[str, LivePosition] = field(default_factory=dict)
    bar_count: int = 0
    initial_equity: float = 0.0
    peak_equity: float = 0.0
    dd_breaker_active: bool = False
    last_signals: dict[str, float] = field(default_factory=dict)


def _engine_to_live_state(engine_state: EngineState) -> LiveState:
    """Convert internal EngineState to LiveState for persistence."""
    live_state = LiveState(
        bar_count=engine_state.bar_count,
        initial_equity=engine_state.initial_equity,
        peak_equity=engine_state.peak_equity,
    )
    for sym, pos in engine_state.positions.items():
        side = "long" if pos.qty > 0 else "short"
        live_state.positions[sym] = LivePosition(
            symbol=sym,
            side=side,
            entry_price=pos.entry_price,
            entry_bar=pos.entry_bar,
            size=Decimal(str(abs(pos.qty))),
            trailing_stop=pos.best_price,  # best_price used as trailing ref
            bars_held=engine_state.bar_count - pos.entry_bar,
        )
    return live_state


def _live_to_engine_state(live_state: LiveState) -> EngineState:
    """Convert persisted LiveState to internal EngineState."""
    engine_state = EngineState(
        bar_count=live_state.bar_count,
        initial_equity=live_state.initial_equity,
        peak_equity=live_state.peak_equity,
    )
    for sym, pos in live_state.positions.items():
        qty = float(pos.size) if pos.side == "long" else -float(pos.size)
        engine_state.positions[sym] = PositionState(
            symbol=sym,
            qty=qty,
            entry_price=pos.entry_price,
            entry_bar=pos.entry_bar,
            best_price=pos.trailing_stop,  # trailing_stop was best_price
        )
    return engine_state


# ── Live engine ──────────────────────────────────────────────────


class LiveEngine:
    """Live CTA trading engine for Hyperliquid.

    Runs the v10g strategy using V10GDecisionEngine for all trade logic.
    """

    MIN_EQUITY = 100.0  # USD
    DEFAULT_LEVERAGE = 5  # cross 5x for all symbols

    def __init__(
        self,
        exchange: ExchangeAdapter,
        *,
        symbols: list[str] | None = None,
        dry_run: bool = False,
        state_file: str = "engine-state.json",
        journal_dir: str = "journal",
        notify: _Notifier | None = None,
        params: V10GStrategyParams | None = None,
    ) -> None:
        self._exchange = exchange
        self._symbols = symbols or V10G_SYMBOLS
        self._dry_run = dry_run
        self._running = False
        self._state = EngineState()
        self._bars_cache: dict[str, pl.DataFrame] = {}
        self._state_file = state_file
        self._journal = TradeJournal(journal_dir)
        self._notify = notify or _NullNotifier()
        self._factor = V10GCompositeFactor()
        self._decision = V10GDecisionEngine(params)

    async def start(self) -> None:
        """Start the live trading loop."""
        logger.info(
            "LiveEngine starting: %d symbols, dry_run=%s",
            len(self._symbols),
            self._dry_run,
        )

        # Preflight checks
        if not await self._preflight():
            logger.error("Preflight checks FAILED — aborting")
            return

        # Try to restore state from disk
        from .state import load_state, save_state

        restored = load_state(self._state_file)
        if restored:
            self._state = _live_to_engine_state(restored)
            account = await self._exchange.get_account_state()
            self._state.peak_equity = max(
                self._state.peak_equity, float(account.equity)
            )
            logger.info(
                "Restored state: bar #%d, %d positions, peak $%.2f",
                self._state.bar_count,
                len(self._state.positions),
                self._state.peak_equity,
            )
            await self._notify.send(
                f"🔄 Engine restarted — restored {len(self._state.positions)} positions, bar #{self._state.bar_count}"
            )
        else:
            await self._notify.send(
                "🚀 Engine started — no prior state, starting fresh"
            )

        self._running = True

        while self._running:
            try:
                await self._wait_for_candle_close()
                await self._tick()
                # Persist state after each tick
                live_state = _engine_to_live_state(self._state)
                save_state(live_state, self._state_file)
            except asyncio.CancelledError:
                logger.info("LiveEngine cancelled")
                live_state = _engine_to_live_state(self._state)
                save_state(live_state, self._state_file)
                break
            except Exception:
                logger.exception("LiveEngine tick error")
                live_state = _engine_to_live_state(self._state)
                save_state(live_state, self._state_file)
                await asyncio.sleep(60)

        logger.info("LiveEngine stopped")

    async def _preflight(self) -> bool:
        """Run preflight checks before trading. Returns True if all pass."""
        logger.info("=== Preflight Checks ===")
        passed = True

        # 1. Exchange connectivity + account state
        try:
            account = await self._exchange.get_account_state()
            logger.info("[✓] Exchange connected")
        except Exception:
            logger.exception("[✗] Exchange connectivity FAILED")
            return False

        # 2. Equity check
        equity = float(account.equity)
        if equity < self.MIN_EQUITY:
            logger.error(
                "[✗] Insufficient equity: $%.2f < $%.2f minimum",
                equity,
                self.MIN_EQUITY,
            )
            return False
        logger.info("[✓] Equity: $%.2f", equity)
        self._state.initial_equity = equity
        self._state.peak_equity = equity

        # 3. Stale positions check
        if account.positions:
            symbols_with_pos = [p.symbol for p in account.positions]
            logger.warning(
                "[!] Found %d existing position(s): %s",
                len(account.positions),
                ", ".join(symbols_with_pos),
            )
            if not self._dry_run:
                logger.info("Closing stale positions before starting...")
                for pos in account.positions:
                    is_buy = pos.size < 0
                    result = await self._exchange.place_market_order(
                        pos.symbol,
                        is_buy,
                        abs(pos.size),
                        reduce_only=True,
                    )
                    if result.success:
                        logger.info("[✓] Closed %s position", pos.symbol)
                    else:
                        logger.error(
                            "[✗] Failed to close %s: %s", pos.symbol, result.message
                        )
                        passed = False
            else:
                logger.info("[DRY RUN] Would close stale positions")
        else:
            logger.info("[✓] No existing positions")

        # 4. Stale orders check
        try:
            open_orders = await self._exchange.get_open_orders()
            if open_orders:
                logger.warning("[!] Found %d stale open order(s)", len(open_orders))
                if not self._dry_run:
                    cancelled = await self._exchange.cancel_all_orders()
                    logger.info("[✓] Cancelled %d stale orders", cancelled)
                else:
                    logger.info(
                        "[DRY RUN] Would cancel %d stale orders", len(open_orders)
                    )
            else:
                logger.info("[✓] No open orders")
        except Exception:
            logger.exception("[✗] Failed to check open orders")
            passed = False

        # 5. Market data spot check
        test_symbol = self._symbols[0] if self._symbols else "BTC"
        try:
            snap = await self._exchange.get_market_snapshot(test_symbol)
            if snap.mid_price > 0:
                logger.info("[✓] Market data OK (%s: $%s)", test_symbol, snap.mid_price)
            else:
                logger.error("[✗] Market data returned zero price for %s", test_symbol)
                passed = False
        except Exception:
            logger.exception("[✗] Market data fetch FAILED for %s", test_symbol)
            passed = False

        # 6. Set leverage for all symbols
        if not self._dry_run:
            for symbol in self._symbols:
                try:
                    await self._exchange.set_leverage(
                        symbol, self.DEFAULT_LEVERAGE, cross=True
                    )
                except Exception:
                    logger.warning("[!] Failed to set leverage for %s", symbol)
            logger.info(
                "[✓] Leverage set to %dx cross for %d symbols",
                self.DEFAULT_LEVERAGE,
                len(self._symbols),
            )
        else:
            logger.info(
                "[DRY RUN] Would set %dx cross leverage for %d symbols",
                self.DEFAULT_LEVERAGE,
                len(self._symbols),
            )

        if passed:
            logger.info("=== Preflight PASSED ===")
        else:
            logger.error("=== Preflight FAILED ===")

        return passed

    async def stop(self) -> None:
        """Stop the trading loop gracefully."""
        logger.info("LiveEngine stopping...")
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def state(self) -> LiveState:
        """Return state in legacy format for external consumers."""
        return _engine_to_live_state(self._state)

    # ── Core loop ────────────────────────────────────────────────

    async def _wait_for_candle_close(self) -> None:
        """Sleep until next 6h candle close (UTC 00:00, 06:00, 12:00, 18:00)."""
        now = datetime.now(tz=UTC)
        hours_since_midnight = now.hour
        next_close_hour = (
            (hours_since_midnight // TIMEFRAME_HOURS) + 1
        ) * TIMEFRAME_HOURS

        if next_close_hour >= 24:
            next_close = now.replace(
                hour=0, minute=0, second=30, microsecond=0
            ) + timedelta(days=1)
        else:
            next_close = now.replace(
                hour=next_close_hour, minute=0, second=30, microsecond=0
            )

        wait_seconds = (next_close - now).total_seconds()
        if wait_seconds > 0:
            logger.info(
                "Waiting %.0f seconds until next candle close (%s)",
                wait_seconds,
                next_close.strftime("%H:%M"),
            )
            await asyncio.sleep(wait_seconds)

    async def _tick(self) -> None:
        """One iteration of the trading loop."""
        logger.info("=== Tick #%d ===", self._state.bar_count + 1)

        # 1. Fetch latest bars
        await self._fetch_bars()

        # 2. Get current equity
        account = await self._exchange.get_account_state()
        equity = float(account.equity)

        # Track returns for vol scaling
        if self._state.peak_equity > 0:
            ret = (equity - self._state.peak_equity) / self._state.peak_equity
            self._state.recent_returns.append(ret)

        # 3. Build snapshots for decision engine
        snapshots = self._build_snapshots()

        # 4. Get decisions from the unified engine
        actions = self._decision.tick(self._state, equity, snapshots)

        # 5. Execute actions
        for action in actions:
            await self._execute_action(action, equity)

        # 6. Journal: record tick equity + signals
        pos_snapshot = {
            sym: {
                "side": "long" if p.qty > 0 else "short",
                "qty": abs(p.qty),
                "entry": p.entry_price,
                "best": p.best_price,
            }
            for sym, p in self._state.positions.items()
        }
        self._journal.record_tick(
            bar=self._state.bar_count,
            equity=equity,
            peak_equity=self._state.peak_equity,
            positions=pos_snapshot,
        )
        self._journal.record_signals(
            bar=self._state.bar_count,
            signals={s: snap.signal for s, snap in snapshots.items()},
        )

        # 7. Summary
        drawdown_pct = (
            (1 - equity / self._state.peak_equity) * 100
            if self._state.peak_equity > 0
            else 0
        )
        pos_summary = (
            ", ".join(
                f"{s} {'L' if p.qty > 0 else 'S'}"
                for s, p in self._state.positions.items()
            )
            or "flat"
        )
        await self._notify.send(
            f"⏰ Tick #{self._state.bar_count} | ${equity:.0f} | "
            f"DD {drawdown_pct:.1f}% | {len(actions)} actions | pos: {pos_summary}"
        )

    # ── Data fetching ────────────────────────────────────────────

    async def _fetch_bars(self) -> None:
        """Fetch latest bars from Binance for all symbols."""
        url = "https://fapi.binance.com/fapi/v1/klines"
        async with httpx.AsyncClient(timeout=30) as client:
            for symbol in self._symbols:
                try:
                    pair = f"{symbol}USDT"
                    resp = await client.get(
                        url,
                        params={"symbol": pair, "interval": "6h", "limit": 200},
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

    # ── Build market snapshots ───────────────────────────────────

    def _build_snapshots(self) -> dict[str, BarSnapshot]:
        """Build BarSnapshot for each symbol from cached bars."""
        # Precompute indicators for all symbols
        indicator_cache: dict[str, dict[str, np.ndarray]] = {}
        for symbol in self._symbols:
            bars = self._bars_cache.get(symbol)
            if bars is None or len(bars) < 50:
                continue
            indicator_cache[symbol] = self._factor.precompute(bars)

        # BTC indicators for regime filter
        btc_ind = indicator_cache.get("BTC")

        snapshots = {}
        for symbol, indicators in indicator_cache.items():
            btc_ref = btc_ind if symbol != "BTC" else None
            sig_array = self._factor.compute_signal_array(indicators, btc_ref)
            sig = float(sig_array[-1]) if len(sig_array) > 0 else 0.0

            c = indicators["close"]
            atr_arr = indicators["atr"]
            rvol_arr = indicators["rvol"]

            snapshots[symbol] = BarSnapshot(
                close=float(c[-1]),
                atr=float(atr_arr[-1]),
                rvol=float(rvol_arr[-1]),
                signal=sig,
            )

        return snapshots

    # ── Action execution ─────────────────────────────────────────

    async def _execute_action(self, action: TradeAction, equity: float) -> None:
        """Execute a single trade action."""
        logger.info(
            "Executing: %s %s qty=%.4f (%s)",
            action.kind,
            action.symbol,
            action.qty,
            action.reason,
        )

        if action.kind == ActionKind.FLATTEN_ALL:
            await self._flatten_all()
            self._running = False
            return

        if action.kind == ActionKind.CLOSE:
            await self._close_position(action.symbol, action.reason)
            return

        if action.kind == ActionKind.PARTIAL_CLOSE:
            await self._partial_close(action.symbol, action.qty, action.reason)
            return

        if action.kind in (ActionKind.OPEN_LONG, ActionKind.OPEN_SHORT):
            await self._open_position(action, equity)
            return

    async def _open_position(self, action: TradeAction, equity: float) -> None:
        """Open a new position."""
        bars = self._bars_cache.get(action.symbol)
        if bars is None:
            return

        current_price = float(bars["close"][-1])
        is_buy = action.kind == ActionKind.OPEN_LONG
        side = "long" if is_buy else "short"
        size = Decimal(str(action.qty))
        size_usd = action.qty * current_price

        logger.info(
            "Opening %s %s: size=$%.0f (%.6f), reason=%s",
            side.upper(),
            action.symbol,
            size_usd,
            action.qty,
            action.reason,
        )

        if self._dry_run:
            logger.info(
                "[DRY RUN] Would place %s %s %.6f @ ~%.2f",
                side,
                action.symbol,
                action.qty,
                current_price,
            )
        else:
            result = await self._exchange.place_market_order(
                action.symbol, is_buy, size
            )
            if not result.success:
                logger.error(
                    "Failed to open %s %s: %s", side, action.symbol, result.message
                )
                return
            if result.avg_price > 0:
                current_price = result.avg_price

        # Update internal state
        qty = action.qty if is_buy else -action.qty
        self._state.positions[action.symbol] = PositionState(
            symbol=action.symbol,
            qty=qty,
            entry_price=current_price,
            entry_bar=self._state.bar_count,
            best_price=current_price,
        )

        await self._notify.send(
            f"📈 OPEN {side.upper()} {action.symbol} | size=${size_usd:.0f} | "
            f"entry={current_price:.2f} | {action.reason}"
        )
        self._journal.record_trade(
            bar=self._state.bar_count,
            kind=action.kind.value,
            symbol=action.symbol,
            qty=action.qty,
            price=current_price,
            reason=action.reason,
            side=side,
        )

    async def _close_position(self, symbol: str, reason: str) -> None:
        """Close an existing position."""
        pos = self._state.positions.get(symbol)
        if pos is None:
            return

        is_buy = pos.qty < 0  # reverse to close
        size = Decimal(str(abs(pos.qty)))
        side = "long" if pos.qty > 0 else "short"

        logger.info(
            "Closing %s %s: size=%.6f, reason=%s",
            side.upper(),
            symbol,
            abs(pos.qty),
            reason,
        )

        if not self._dry_run:
            result = await self._exchange.place_market_order(
                symbol, is_buy, size, reduce_only=True
            )
            if not result.success:
                logger.error("Failed to close %s: %s", symbol, result.message)
                return

        del self._state.positions[symbol]
        held = self._state.bar_count - pos.entry_bar

        # Compute PnL
        exit_price = (
            float(self._bars_cache.get(symbol, pl.DataFrame())["close"][-1])
            if symbol in self._bars_cache
            else pos.entry_price
        )
        if pos.qty > 0:
            pnl = (exit_price - pos.entry_price) * abs(pos.qty)
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        else:
            pnl = (pos.entry_price - exit_price) * abs(pos.qty)
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price * 100

        await self._notify.send(
            f"📉 CLOSE {side.upper()} {symbol} | held={held} bars | "
            f"pnl=${pnl:.2f} ({pnl_pct:+.1f}%) | {reason}"
        )
        self._journal.record_trade(
            bar=self._state.bar_count,
            kind="close",
            symbol=symbol,
            qty=abs(pos.qty),
            price=exit_price,
            reason=reason,
            side=side,
            entry_price=pos.entry_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            held_bars=held,
        )

    async def _partial_close(self, symbol: str, qty: float, reason: str) -> None:
        """Partial close a position."""
        pos = self._state.positions.get(symbol)
        if pos is None:
            return

        is_buy = pos.qty < 0
        size = Decimal(str(qty))
        side = "long" if pos.qty > 0 else "short"

        logger.info(
            "Partial close %s %s: qty=%.6f, reason=%s",
            side.upper(),
            symbol,
            qty,
            reason,
        )

        if not self._dry_run:
            result = await self._exchange.place_market_order(
                symbol, is_buy, size, reduce_only=True
            )
            if not result.success:
                logger.error("Failed to partial close %s: %s", symbol, result.message)
                return

        # Update position qty
        if pos.qty > 0:
            pos.qty -= qty
        else:
            pos.qty += qty
        pos.partial_taken = True

        await self._notify.send(
            f"📊 PARTIAL {side.upper()} {symbol} | closed {qty:.6f} | {reason}"
        )
        exit_price = (
            float(self._bars_cache.get(symbol, pl.DataFrame())["close"][-1])
            if symbol in self._bars_cache
            else pos.entry_price
        )
        if pos.qty > 0:
            pnl = (exit_price - pos.entry_price) * qty
        else:
            pnl = (pos.entry_price - exit_price) * qty
        self._journal.record_trade(
            bar=self._state.bar_count,
            kind="partial_close",
            symbol=symbol,
            qty=qty,
            price=exit_price,
            reason=reason,
            side=side,
            entry_price=pos.entry_price,
            pnl=pnl,
            pnl_pct=(pnl / (pos.entry_price * qty) * 100) if pos.entry_price > 0 else 0,
            held_bars=self._state.bar_count - pos.entry_bar,
        )

    async def _flatten_all(self) -> None:
        """Emergency: close all positions."""
        logger.warning("FLATTENING ALL POSITIONS")
        await self._notify.send("🚨 MAX DRAWDOWN — FLATTENING ALL POSITIONS")
        symbols = list(self._state.positions.keys())
        for symbol in symbols:
            await self._close_position(symbol, "flatten_all")
        if not self._dry_run:
            await self._exchange.cancel_all_orders()
