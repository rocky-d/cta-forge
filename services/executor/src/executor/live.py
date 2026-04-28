"""Live execution engine for Hyperliquid.

Runs the v10g CTA strategy on real market data:
- Every 6h (aligned to candle close): fetch bars → compute signals → decide → execute
- Uses V10GDecisionEngine for all trade decisions (shared with backtest)
- Uses exchange adapter for all HL interactions
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import httpx
import numpy as np
import polars as pl
from data_service.fetcher import fetch_klines
from data_service.store import ParquetStore

if TYPE_CHECKING:
    from exchange.adapter import AccountState, ExchangeAdapter

from alpha_service.factors.v10g_composite import (
    V10GCompositeFactor,
    V10GCompositeParams,
)
from core.constants import (
    DEFAULT_TIMEFRAME,
    V10G_SYMBOLS,
    V10G_TESTNET_EXCLUDED,
)

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
from .notify import NullNotifier, _Notifier
from .targeting import TargetOrder, TargetWeightStrategy, weights_to_orders

logger = logging.getLogger(__name__)

V10G_PROFILE_SLUG = "v10g-engine-6h"
V16A_PROFILE_SLUG = "v16a-badscore-overlay"


# ── v10g strategy defaults ───────────────────────────────────────

# Note: TIMEFRAME_HOURS is now dynamic via self._decision.params.timeframe_hours


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
        data_dir: str = "data",
        notify: _Notifier | None = None,
        params: V10GStrategyParams | None = None,
        clean_start: bool = False,
        strategy_profile: str = V10G_PROFILE_SLUG,
        target_strategy: TargetWeightStrategy | None = None,
        min_order_notional: float = 10.0,
    ) -> None:
        self._exchange = exchange
        # Auto-filter symbols unavailable on testnet
        is_testnet = os.getenv("HL_NETWORK", "testnet") == "testnet"
        all_symbols = symbols or list(V10G_SYMBOLS)
        if is_testnet:
            all_symbols = [s for s in all_symbols if s not in V10G_TESTNET_EXCLUDED]
        self._symbols = all_symbols
        self._dry_run = dry_run
        self._running = False
        self._clean_start = clean_start
        self._state = EngineState()
        self._bars_cache: dict[str, pl.DataFrame] = {}
        self._state_file = state_file
        self._store = ParquetStore(data_dir)
        self._journal = TradeJournal(journal_dir)
        self._notify = notify or NullNotifier()
        self._target_strategy = target_strategy
        self._strategy_profile = (
            target_strategy.profile.slug
            if target_strategy is not None
            else strategy_profile
        )
        self._min_order_notional = min_order_notional
        if target_strategy is None and self._strategy_profile != V10G_PROFILE_SLUG:
            msg = (
                f"Strategy profile {self._strategy_profile!r} has no live target "
                "provider wired yet"
            )
            raise ValueError(msg)
        params = params or V10GStrategyParams()
        # Pass params to factor and decision engine
        self._factor = V10GCompositeFactor(
            params=V10GCompositeParams(
                timeframe_hours=params.timeframe_hours,
                mom_lookbacks=params.mom_lookbacks,
                adx_threshold=params.adx_threshold,
                adx_ensemble=params.adx_ensemble,
                signal_persistence=params.signal_persistence,
                donchian_period=params.donchian_period,
                rvol_lookback=params.rvol_lookback,
            )
        )
        self._decision = V10GDecisionEngine(params)

    async def start(self) -> None:
        """Start the live trading loop."""
        logger.info(
            "LiveEngine starting: profile=%s, symbols=%d, dry_run=%s",
            self._strategy_profile,
            len(self._symbols),
            self._dry_run,
        )

        # Preflight checks
        if not await self._preflight():
            logger.error("Preflight checks FAILED — aborting")
            return

        # Try to restore state from disk, then reconcile with exchange
        from .state import load_state, save_state

        restored = load_state(self._state_file)
        account = await self._exchange.get_account_state()
        exchange_positions = {p.symbol: p for p in account.positions}

        if restored and not self._clean_start:
            self._state = _live_to_engine_state(restored)
            self._state.peak_equity = max(
                self._state.peak_equity, float(account.equity)
            )

            # Reconcile state positions with exchange
            state_syms = set(self._state.positions.keys())
            exchange_syms = set(exchange_positions.keys())

            # Positions in state but not on exchange → already closed externally
            for sym in state_syms - exchange_syms:
                logger.warning(
                    "[reconcile] %s in state but not on exchange — removing", sym
                )
                del self._state.positions[sym]

            # Positions on exchange but not in state → unexpected, adopt them
            for sym in exchange_syms - state_syms:
                pos = exchange_positions[sym]
                logger.warning(
                    "[reconcile] %s on exchange but not in state — adopting "
                    "(side=%s, size=%s, entry=$%s)",
                    sym,
                    "long" if pos.size > 0 else "short",
                    abs(pos.size),
                    pos.entry_price,
                )
                self._state.positions[sym] = PositionState(
                    symbol=sym,
                    qty=float(pos.size),
                    entry_price=float(pos.entry_price),
                    entry_bar=max(self._state.bar_count - 1, 0),
                    best_price=float(pos.entry_price),
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
        elif exchange_positions and not self._clean_start:
            # No state file but exchange has positions → adopt all
            logger.warning(
                "[reconcile] No state file, adopting %d exchange position(s)",
                len(exchange_positions),
            )
            self._state.initial_equity = float(account.equity)
            self._state.peak_equity = float(account.equity)
            for sym, pos in exchange_positions.items():
                self._state.positions[sym] = PositionState(
                    symbol=sym,
                    qty=float(pos.size),
                    entry_price=float(pos.entry_price),
                    entry_bar=0,
                    best_price=float(pos.entry_price),
                )
            await self._notify.send(
                f"🔄 Engine restarted — adopted {len(exchange_positions)} orphan position(s)"
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

        # 3. Check existing positions (reconcile later in start())
        if account.positions:
            symbols_with_pos = [p.symbol for p in account.positions]
            if self._clean_start:
                logger.warning(
                    "[!] CLEAN_START: closing %d position(s): %s",
                    len(account.positions),
                    ", ".join(symbols_with_pos),
                )
                if not self._dry_run:
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
                                "[✗] Failed to close %s: %s",
                                pos.symbol,
                                result.message,
                            )
                            passed = False
                else:
                    logger.info("[DRY RUN] Would close positions")
            else:
                logger.info(
                    "[i] Found %d existing position(s): %s — will reconcile after state load",
                    len(account.positions),
                    ", ".join(symbols_with_pos),
                )
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
        """Sleep until next candle close (e.g. UTC 00:00, 06:00, 12:00, 18:00 for 6h)."""
        now = datetime.now(tz=UTC)
        hours_since_midnight = now.hour
        tf_hours = (
            self._target_strategy.profile.timeframe_hours
            if self._target_strategy is not None
            else self._decision.p.timeframe_hours
        )
        next_close_hour = ((hours_since_midnight // tf_hours) + 1) * tf_hours

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

        if self._target_strategy is None:
            # 1. Fetch latest bars for the v10g decision engine.
            await self._fetch_bars()

        # 2. Get current equity
        account = await self._exchange.get_account_state()
        equity = float(account.equity)

        # Track returns for vol scaling
        if self._state.peak_equity > 0:
            ret = (equity - self._state.peak_equity) / self._state.peak_equity
            self._state.recent_returns.append(ret)

        snapshots: dict[str, BarSnapshot] = {}
        action_count = 0
        if self._target_strategy is not None:
            orders = await self._execute_target_portfolio(account, equity)
            action_count = len(orders)
            self._state.bar_count += 1
        else:
            # 3. Build snapshots for decision engine
            snapshots = self._build_snapshots()

            # 4. Snapshot positions BEFORE tick: tick() deletes closed positions
            # from state internally (so opens can reuse the slot on the same bar).
            # We need the pre-tick snapshot to execute closes and partials.
            positions_before: dict[str, PositionState] = {
                sym: PositionState(
                    symbol=pos.symbol,
                    qty=pos.qty,
                    entry_price=pos.entry_price,
                    entry_bar=pos.entry_bar,
                    best_price=pos.best_price,
                    partial_taken=pos.partial_taken,
                )
                for sym, pos in self._state.positions.items()
            }

            # 5. Get decisions from the unified engine
            actions = self._decision.tick(self._state, equity, snapshots)
            action_count = len(actions)

            # 6. Execute actions (pass pre-tick snapshot for close/partial settlement)
            for action in actions:
                await self._execute_action(action, equity, positions_before)

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
            f"DD {drawdown_pct:.1f}% | {action_count} actions | pos: {pos_summary}"
        )

    # ── Target portfolio execution ───────────────────────────────

    @staticmethod
    def _normalize_live_symbol(symbol: str) -> str:
        """Convert research symbols like BTCUSDT to live exchange symbols like BTC."""
        return symbol[:-4] if symbol.endswith("USDT") else symbol

    def _normalize_target_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """Normalize target symbols and only allow configured live universe opens."""
        normalized: dict[str, float] = {}
        allowed = set(self._symbols)
        for raw_symbol, weight in weights.items():
            symbol = self._normalize_live_symbol(raw_symbol)
            if symbol not in allowed:
                logger.warning(
                    "Ignoring target for %s outside configured universe", raw_symbol
                )
                continue
            normalized[symbol] = normalized.get(symbol, 0.0) + float(weight)
        return normalized

    def _sync_target_state_from_account(self, account: "AccountState") -> None:
        """Use exchange account positions as source of truth before target orders."""
        next_positions: dict[str, PositionState] = {}
        for pos in account.positions:
            size = float(pos.size)
            if abs(size) <= 1e-12:
                continue
            existing = self._state.positions.get(pos.symbol)
            next_positions[pos.symbol] = PositionState(
                symbol=pos.symbol,
                qty=size,
                entry_price=float(pos.entry_price),
                entry_bar=existing.entry_bar if existing else self._state.bar_count,
                best_price=existing.best_price if existing else float(pos.entry_price),
                partial_taken=existing.partial_taken if existing else False,
            )
        self._state.positions = next_positions

    async def _fetch_target_prices(self, symbols: set[str]) -> dict[str, float]:
        """Fetch live mark prices for target reconciliation."""
        prices: dict[str, float] = {}
        for symbol in sorted(symbols):
            try:
                snap = await self._exchange.get_market_snapshot(symbol)
            except Exception:
                logger.exception("Failed to fetch target price for %s", symbol)
                continue
            price = float(snap.mark_price or snap.mid_price)
            if price > 0:
                prices[symbol] = price
        return prices

    async def _execute_target_portfolio(
        self, account: "AccountState", equity: float
    ) -> list[TargetOrder]:
        """Reconcile a target-weight strategy into market-order deltas."""
        if self._target_strategy is None:
            return []

        self._sync_target_state_from_account(account)
        target = self._target_strategy.target(datetime.now(tz=UTC)).capped()
        target_weights = self._normalize_target_weights(dict(target.weights))
        positions = {pos.symbol: float(pos.size) for pos in account.positions}
        symbols = set(positions) | set(target_weights)
        prices = await self._fetch_target_prices(symbols)
        orders = weights_to_orders(
            positions,
            prices,
            equity,
            target_weights,
            min_notional=self._min_order_notional,
        )

        for order in orders:
            price = prices.get(order.symbol)
            if price is None:
                continue
            await self._execute_target_order(order, price)

        logger.info(
            "Target profile %s produced %d order(s), gross=%.3f",
            self._strategy_profile,
            len(orders),
            target.gross,
        )
        return orders

    async def _execute_target_order(self, order: TargetOrder, price: float) -> None:
        """Execute one target reconciliation order and update local state."""
        is_buy = order.side == "buy"
        size = Decimal(str(order.qty))
        logger.info(
            "Target order: %s %s qty=%.6f reduce_only=%s current=%.3f target=%.3f",
            order.side.upper(),
            order.symbol,
            order.qty,
            order.reduce_only,
            order.current_weight,
            order.target_weight,
        )

        fill_price = price
        if self._dry_run:
            logger.info(
                "[DRY RUN] Would target-order %s %s %.6f @ ~%.2f reduce_only=%s",
                order.side,
                order.symbol,
                order.qty,
                price,
                order.reduce_only,
            )
        else:
            result = await self._exchange.place_market_order(
                order.symbol,
                is_buy,
                size,
                reduce_only=order.reduce_only,
            )
            if not result.success:
                logger.error("Failed target order %s: %s", order.symbol, result.message)
                return
            if result.avg_price > 0:
                fill_price = result.avg_price

        self._apply_target_fill(order, fill_price)
        self._journal.record_trade(
            bar=self._state.bar_count,
            kind="target_buy" if is_buy else "target_sell",
            symbol=order.symbol,
            qty=order.qty,
            price=fill_price,
            reason=f"target:{self._strategy_profile}",
            side="long" if is_buy else "short",
        )

    def _apply_target_fill(self, order: TargetOrder, price: float) -> None:
        """Apply a successfully executed target order to local engine state."""
        signed_qty = order.qty if order.side == "buy" else -order.qty
        current = self._state.positions.get(order.symbol)
        old_qty = current.qty if current is not None else 0.0
        new_qty = old_qty + signed_qty

        if abs(new_qty) <= 1e-12 or (order.reduce_only and old_qty * new_qty <= 0):
            self._state.positions.pop(order.symbol, None)
            return

        if current is None or old_qty * new_qty <= 0:
            entry_price = price
            entry_bar = self._state.bar_count
            best_price = price
            partial_taken = False
        elif abs(new_qty) > abs(old_qty):
            added_qty = abs(signed_qty)
            entry_price = (
                abs(old_qty) * current.entry_price + added_qty * price
            ) / abs(new_qty)
            entry_bar = current.entry_bar
            best_price = current.best_price
            partial_taken = current.partial_taken
        else:
            entry_price = current.entry_price
            entry_bar = current.entry_bar
            best_price = current.best_price
            partial_taken = current.partial_taken

        self._state.positions[order.symbol] = PositionState(
            symbol=order.symbol,
            qty=new_qty,
            entry_price=entry_price,
            entry_bar=entry_bar,
            best_price=best_price,
            partial_taken=partial_taken,
        )

    # ── Data fetching ────────────────────────────────────────────

    async def _fetch_bars(self) -> None:
        """Fetch bars from local cache (parquet), backfill from Binance as needed."""
        interval = self._decision.p.timeframe_str or DEFAULT_TIMEFRAME
        min_bars = 200  # strategy warm-up requirement

        async def _fetch_symbol(
            client: httpx.AsyncClient, symbol: str
        ) -> tuple[str, pl.DataFrame | None]:
            """Fetch one symbol, return (symbol, df) or (symbol, None)."""
            pair = f"{symbol}USDT"
            try:
                # 1. Check local parquet
                latest = self._store.latest_timestamp(pair, interval)
                if latest is not None:
                    age_hours = (datetime.now(tz=UTC) - latest).total_seconds() / 3600
                    need_fetch = age_hours > self._decision.p.timeframe_hours
                else:
                    need_fetch = True

                if need_fetch:
                    start_ms = int(latest.timestamp() * 1000) + 1 if latest else None
                    new_bars = await fetch_klines(
                        client,
                        symbol=pair,
                        interval=interval,
                        start_ms=start_ms,
                        limit=1000,
                    )
                    if not new_bars.is_empty():
                        self._store.write(pair, interval, new_bars)
                        logger.info("Stored %d new bars for %s", len(new_bars), pair)

                # 2. Read from store
                df = self._store.read(pair, interval)
                if df.is_empty():
                    logger.warning("No data available for %s", pair)
                    return symbol, None

                if len(df) > min_bars:
                    df = df.tail(min_bars)

                return symbol, df

            except Exception:
                logger.exception("Error fetching %s", pair)
                return symbol, None

        # Fetch all symbols concurrently
        async with httpx.AsyncClient(timeout=30) as client:
            results = await asyncio.gather(
                *(_fetch_symbol(client, sym) for sym in self._symbols)
            )

        for symbol, df in results:
            if df is not None:
                self._bars_cache[symbol] = df

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

    async def _execute_action(
        self,
        action: TradeAction,
        equity: float,
        positions_before: dict[str, PositionState],
    ) -> None:
        """Execute a single trade action."""
        logger.info(
            "Executing: %s %s qty=%.4f (%s)",
            action.kind,
            action.symbol,
            action.qty,
            action.reason,
        )

        if action.kind == ActionKind.FLATTEN_ALL:
            await self._flatten_all(positions_before)
            self._running = False
            return

        if action.kind == ActionKind.CLOSE:
            await self._close_position(action.symbol, action.reason, positions_before)
            return

        if action.kind == ActionKind.PARTIAL_CLOSE:
            await self._partial_close(
                action.symbol, action.qty, action.reason, positions_before
            )
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

    async def _close_position(
        self,
        symbol: str,
        reason: str,
        positions_before: dict[str, PositionState] | None = None,
    ) -> None:
        """Close an existing position.

        Args:
            positions_before: pre-tick snapshot. tick() deletes closed positions
                from state internally, so we must use the snapshot to know what
                to close. Falls back to current state for _flatten_all calls.
        """
        pos = (
            positions_before.get(symbol)
            if positions_before is not None
            else self._state.positions.get(symbol)
        )
        if pos is None:
            return

        is_buy = bool(pos.qty < 0)  # reverse to close
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

        # Remove from state if tick() didn't already
        self._state.positions.pop(symbol, None)
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

    async def _partial_close(
        self,
        symbol: str,
        qty: float,
        reason: str,
        positions_before: dict[str, PositionState] | None = None,
    ) -> None:
        """Partial close a position.

        tick() already adjusted pos.qty in state, so we use positions_before
        to get the original side for determining buy/sell direction.
        """
        pos_before = (
            positions_before.get(symbol)
            if positions_before is not None
            else self._state.positions.get(symbol)
        )
        if pos_before is None:
            return

        is_buy = bool(pos_before.qty < 0)
        size = Decimal(str(qty))
        side = "long" if pos_before.qty > 0 else "short"

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

        # tick() already adjusted pos.qty and set partial_taken in state;
        # no need to update here.

        await self._notify.send(
            f"📊 PARTIAL {side.upper()} {symbol} | closed {qty:.6f} | {reason}"
        )
        exit_price = (
            float(self._bars_cache.get(symbol, pl.DataFrame())["close"][-1])
            if symbol in self._bars_cache
            else pos_before.entry_price
        )
        if pos_before.qty > 0:
            pnl = (exit_price - pos_before.entry_price) * qty
        else:
            pnl = (pos_before.entry_price - exit_price) * qty
        self._journal.record_trade(
            bar=self._state.bar_count,
            kind="partial_close",
            symbol=symbol,
            qty=qty,
            price=exit_price,
            reason=reason,
            side=side,
            entry_price=pos_before.entry_price,
            pnl=pnl,
            pnl_pct=(pnl / (pos_before.entry_price * qty) * 100)
            if pos_before.entry_price > 0
            else 0,
            held_bars=self._state.bar_count - pos_before.entry_bar,
        )

    async def _flatten_all(
        self, positions_before: dict[str, PositionState] | None = None
    ) -> None:
        """Emergency: close all positions."""
        logger.warning("FLATTENING ALL POSITIONS")
        await self._notify.send("🚨 MAX DRAWDOWN — FLATTENING ALL POSITIONS")
        # Use pre-tick snapshot: tick() already deleted positions from state
        source = (
            positions_before if positions_before is not None else self._state.positions
        )
        symbols = list(source.keys())
        for symbol in symbols:
            await self._close_position(symbol, "flatten_all", positions_before)
        if not self._dry_run:
            await self._exchange.cancel_all_orders()
