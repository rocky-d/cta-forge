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

import numpy as np
import polars as pl
from data_service.store import ParquetStore

if TYPE_CHECKING:
    from exchange.adapter import AccountState, ExchangeAdapter

    from .state import LiveStateStore

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
from .journal import LiveJournalStore, TradeJournal
from .live_data import fetch_live_bars
from .live_target import execute_target_portfolio, sync_target_state_from_account
from .notify import NullNotifier, _Notifier
from .targeting import TargetOrder, TargetWeightStrategy

logger = logging.getLogger(__name__)

V10G_PROFILE_SLUG = "v10g-engine-6h"
V16A_PROFILE_SLUG = "v16a-badscore-overlay"


def _format_usd(value: float) -> str:
    """Format a USD notional compactly for chat notifications."""
    amount = abs(float(value))
    if amount >= 10:
        return f"${amount:.0f}"
    return f"${amount:.1f}"


def _summarize_target_orders(orders: list[TargetOrder], *, limit: int = 5) -> str:
    """Return a readable target-order block for tick notifications."""
    if not orders:
        return "Actions: none"
    lines = [f"Actions ({len(orders)}):"]
    for order in orders[:limit]:
        reduce = " reduce" if order.reduce_only else ""
        lines.append(
            f"- {order.side.upper()} {order.symbol} "
            f"{_format_usd(order.delta_notional)}{reduce}"
        )
    if len(orders) > limit:
        lines.append(f"- +{len(orders) - limit} more")
    return "\n".join(lines)


def _summarize_trade_actions(actions: list[TradeAction], *, limit: int = 5) -> str:
    """Return a readable v10g action block for tick notifications."""
    if not actions:
        return "Actions: none"
    lines = [f"Actions ({len(actions)}):"]
    for action in actions[:limit]:
        lines.append(f"- {action.kind.value} {action.symbol}")
    if len(actions) > limit:
        lines.append(f"- +{len(actions) - limit} more")
    return "\n".join(lines)


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
    recent_returns: list[float] = field(default_factory=list)
    last_tick_equity: float | None = None


def _engine_to_live_state(
    engine_state: EngineState,
    *,
    last_tick_equity: float | None = None,
) -> LiveState:
    """Convert internal EngineState to LiveState for persistence."""
    live_state = LiveState(
        bar_count=engine_state.bar_count,
        initial_equity=engine_state.initial_equity,
        peak_equity=engine_state.peak_equity,
        recent_returns=list(engine_state.recent_returns[-120:]),
        last_tick_equity=last_tick_equity,
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
        recent_returns=list(live_state.recent_returns[-120:]),
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
        max_order_notional: float | None = None,
        min_equity: float | None = None,
        min_available_balance: float | None = None,
        max_equity: float | None = None,
        leverage: int | None = None,
        live_instance_id: str | None = None,
        run_id: str | None = None,
        public_instance_slug: str | None = None,
        journal: LiveJournalStore | None = None,
        state_store: LiveStateStore | None = None,
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
        self._last_tick_equity: float | None = None
        self._bars_cache: dict[str, pl.DataFrame] = {}
        self._state_file = state_file
        self._store = ParquetStore(data_dir)
        self._state_store = state_store
        if journal is None:
            journal = TradeJournal(
                journal_dir,
                live_instance_id=live_instance_id,
                run_id=run_id,
                public_instance_slug=public_instance_slug,
            )
        self._journal: LiveJournalStore = journal
        self._notify = notify or NullNotifier()
        self._target_strategy = target_strategy
        self._strategy_profile = (
            target_strategy.profile.slug
            if target_strategy is not None
            else strategy_profile
        )
        self._min_order_notional = min_order_notional
        self._max_order_notional = max_order_notional
        self._min_equity = min_equity or self.MIN_EQUITY
        self._min_available_balance = min_available_balance
        self._max_equity = max_equity
        self._leverage = leverage or self.DEFAULT_LEVERAGE
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

        # Try to restore state from disk, then reconcile with exchange.
        # Keep this import local: state.py imports LiveState/LivePosition from this
        # module for backward-compatible persistence, so a top-level import would
        # create a circular dependency.
        from .state import JsonFileLiveStateStore

        state_store = self._state_store
        if state_store is None:
            state_store = JsonFileLiveStateStore(self._state_file)
        restored = state_store.load()
        account = await self._exchange.get_account_state()
        exchange_positions = {p.symbol: p for p in account.positions}

        if restored and not self._clean_start:
            self._state = _live_to_engine_state(restored)
            self._last_tick_equity = restored.last_tick_equity
            self._restore_equity_state_from_journal(float(account.equity))

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
            self._restore_equity_state_from_journal(float(account.equity))
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
            if not self._clean_start:
                self._restore_equity_state_from_journal(float(account.equity))
            await self._notify.send(
                "🚀 Engine started — no prior state, starting fresh"
            )

        self._running = True

        while self._running:
            try:
                await self._wait_for_candle_close()
                await self._tick()
                # Persist state after each tick
                live_state = _engine_to_live_state(
                    self._state,
                    last_tick_equity=self._last_tick_equity,
                )
                state_store.save(live_state)
            except asyncio.CancelledError:
                logger.info("LiveEngine cancelled")
                live_state = _engine_to_live_state(
                    self._state,
                    last_tick_equity=self._last_tick_equity,
                )
                state_store.save(live_state)
                break
            except Exception:
                logger.exception("LiveEngine tick error")
                live_state = _engine_to_live_state(
                    self._state,
                    last_tick_equity=self._last_tick_equity,
                )
                state_store.save(live_state)
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
        if equity < self._min_equity:
            logger.error(
                "[✗] Insufficient equity: $%.2f < $%.2f minimum",
                equity,
                self._min_equity,
            )
            return False
        if self._max_equity is not None and equity > self._max_equity:
            logger.error(
                "[✗] Equity cap exceeded: $%.2f > $%.2f maximum",
                equity,
                self._max_equity,
            )
            return False
        available_balance = float(account.available_balance)
        if (
            self._min_available_balance is not None
            and available_balance < self._min_available_balance
        ):
            logger.error(
                "[✗] Insufficient available balance: $%.2f < $%.2f minimum",
                available_balance,
                self._min_available_balance,
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
                        symbol, self._leverage, cross=True
                    )
                except Exception:
                    logger.warning("[!] Failed to set leverage for %s", symbol)
            logger.info(
                "[✓] Leverage set to %dx cross for %d symbols",
                self._leverage,
                len(self._symbols),
            )
        else:
            logger.info(
                "[DRY RUN] Would set %dx cross leverage for %d symbols",
                self._leverage,
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
        return _engine_to_live_state(
            self._state,
            last_tick_equity=self._last_tick_equity,
        )

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

    def _record_tick_return(self, equity: float) -> None:
        """Append the realized tick-to-tick equity return for vol scaling."""
        if self._last_tick_equity is not None and self._last_tick_equity > 0:
            ret = (equity - self._last_tick_equity) / self._last_tick_equity
            self._state.recent_returns.append(ret)
            if len(self._state.recent_returns) > 120:
                self._state.recent_returns.pop(0)
        self._last_tick_equity = equity

    def _restore_equity_state_from_journal(self, account_equity: float) -> None:
        """Recover persisted equity-derived state from the append-only journal.

        The state file is the primary checkpoint, but older target-mode builds
        could persist a stale peak. The journal records every observed equity,
        so use it as a defensive high-water-mark source across deploy restarts.
        """
        records = self._journal.load_equity()
        observed_equities = [float(account_equity)]
        for record in records:
            for key in ("equity", "peak"):
                value = record.get(key)
                if value is not None:
                    observed_equities.append(float(value))
        self._state.peak_equity = max(self._state.peak_equity, *observed_equities)
        if self._last_tick_equity is None and records:
            last_equity = records[-1].get("equity")
            if last_equity is not None:
                self._last_tick_equity = float(last_equity)

    async def _position_snapshot(self, equity: float) -> dict[str, dict]:
        """Build the actual-position journal snapshot for the current state.

        The journal keeps private runtime fields for persistence/reconciliation,
        plus a public-safe signed exposure weight used by dashboard collectors.
        """
        equity_value = float(equity)
        positions: dict[str, dict] = {}
        for sym, pos in self._state.positions.items():
            side = "long" if pos.qty > 0 else "short"
            record = {
                "side": side,
                "qty": abs(pos.qty),
                "entry": pos.entry_price,
                "best": pos.best_price,
            }
            try:
                snap = await self._exchange.get_market_snapshot(sym)
                price = float(snap.mark_price or snap.mid_price)
            except Exception:
                logger.warning("Failed to price position %s for public weight", sym)
            else:
                if equity_value > 0 and price > 0:
                    record["weight"] = float(pos.qty * price / equity_value)
            positions[sym] = record
        return positions

    async def _tick(self) -> None:
        """One iteration of the trading loop."""
        logger.info("=== Tick #%d ===", self._state.bar_count + 1)

        if self._target_strategy is None:
            # 1. Fetch latest bars for the v10g decision engine.
            await self._fetch_bars()
        else:
            # Keep the parquet cache fresh for target providers that build
            # targets from local market data, then force any cached target set
            # to observe the newly written parquet before this tick trades.
            await self._fetch_target_data()
            refresh = getattr(self._target_strategy, "refresh", None)
            if refresh is not None:
                refresh(force=True)

        # 2. Get current equity
        account = await self._exchange.get_account_state()
        equity = float(account.equity)

        # Track realized tick-to-tick returns for vol scaling. Drawdown/peak
        # updates happen below; returns must not be measured against peak equity.
        self._record_tick_return(equity)

        snapshots: dict[str, BarSnapshot] = {}
        action_summary = "0 actions"
        if self._target_strategy is not None:
            if await self._handle_target_drawdown_stop(account, equity):
                action_summary = "max drawdown flatten"
            else:
                orders = await self._execute_target_portfolio(account, equity)
                action_summary = _summarize_target_orders(orders)
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
            action_summary = _summarize_trade_actions(actions)

            # 6. Execute actions (pass pre-tick snapshot for close/partial settlement)
            for action in actions:
                if not await self._execute_action(action, equity, positions_before):
                    logger.error(
                        "Stopping action execution after failed %s", action.kind
                    )
                    break

        # Target-weight strategies do not enter V10GDecisionEngine.tick(), so
        # keep live equity invariants here for both execution modes.
        self._state.peak_equity = max(self._state.peak_equity, equity)

        # 6. Journal: record tick equity + signals
        pos_snapshot = await self._position_snapshot(equity)
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
            max(0.0, (1 - equity / self._state.peak_equity) * 100)
            if self._state.peak_equity > 0
            else 0.0
        )
        pos_summary = (
            ", ".join(
                f"{s} {'L' if p.qty > 0 else 'S'}"
                for s, p in self._state.positions.items()
            )
            or "flat"
        )
        cash_summary = f" | Avail ${float(account.available_balance):.1f}"
        if account.unrealized_pnl is not None:
            cash_summary += f" | uPnL ${float(account.unrealized_pnl):+.1f}"
        await self._notify.send(
            f"⏰ Tick #{self._state.bar_count} | Eq ${equity:.1f}"
            f"{cash_summary} | DD {drawdown_pct:.2f}%\n"
            f"{action_summary}\n"
            f"Positions: {pos_summary}"
        )

    # ── Target portfolio execution ───────────────────────────────

    async def _handle_target_drawdown_stop(
        self, account: "AccountState", equity: float
    ) -> bool:
        """Fail closed for target-mode profiles when max drawdown is breached."""
        peak = max(self._state.peak_equity, equity)
        cur_dd = (peak - equity) / peak if peak > 0 else 0.0
        if cur_dd < self._decision.p.max_drawdown:
            return False

        logger.warning(
            "TARGET MAX DRAWDOWN %.1f%% >= %.1f%% — flatten all",
            cur_dd * 100,
            self._decision.p.max_drawdown * 100,
        )
        self._state.dd_breaker_active = True
        sync_target_state_from_account(self._state, account, set(self._symbols))
        if self._state.positions:
            positions_before = {
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
            ok = await self._flatten_all(positions_before)
            if not ok:
                await self._notify.send(
                    f"🚨 TARGET MAX DD {cur_dd * 100:.1f}% — flatten failed; "
                    "new exposure suppressed"
                )
                return True
        elif not self._dry_run:
            await self._exchange.cancel_all_orders()
        await self._notify.send(
            f"🛑 TARGET MAX DD {cur_dd * 100:.1f}% — new exposure suppressed"
        )
        return True

    async def _execute_target_portfolio(
        self, account: "AccountState", equity: float
    ) -> list[TargetOrder]:
        """Reconcile a target-weight strategy into market-order deltas."""
        return await execute_target_portfolio(
            exchange=self._exchange,
            journal=self._journal,
            state=self._state,
            account=account,
            equity=equity,
            target_strategy=self._target_strategy,
            symbols=self._symbols,
            profile=self._strategy_profile,
            dry_run=self._dry_run,
            min_order_notional=self._min_order_notional,
            max_order_notional=self._max_order_notional,
        )

    # ── Data fetching ────────────────────────────────────────────

    async def _fetch_target_data(self) -> None:
        """Refresh data intervals requested by the target strategy."""
        if self._target_strategy is None:
            return
        required = getattr(self._target_strategy, "required_timeframes", ())
        for spec in required:
            interval, timeframe_hours, *rest = spec
            min_bars = int(rest[0]) if rest else 200
            await self._fetch_bars(
                interval=interval,
                timeframe_hours=timeframe_hours,
                min_bars=min_bars,
            )

    async def _fetch_bars(
        self,
        *,
        interval: str | None = None,
        timeframe_hours: int | None = None,
        min_bars: int = 200,
    ) -> None:
        """Fetch bars from local cache (parquet), backfill from Binance as needed."""
        interval = interval or self._decision.p.timeframe_str or DEFAULT_TIMEFRAME
        timeframe_hours = timeframe_hours or self._decision.p.timeframe_hours
        self._bars_cache.update(
            await fetch_live_bars(
                store=self._store,
                symbols=self._symbols,
                interval=interval,
                timeframe_hours=timeframe_hours,
                min_bars=min_bars,
            )
        )

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
    ) -> bool:
        """Execute a single trade action; return whether execution succeeded."""
        logger.info(
            "Executing: %s %s qty=%.4f (%s)",
            action.kind,
            action.symbol,
            action.qty,
            action.reason,
        )

        if action.kind == ActionKind.FLATTEN_ALL:
            ok = await self._flatten_all(positions_before)
            self._running = False
            return ok

        if action.kind == ActionKind.CLOSE:
            return await self._close_position(
                action.symbol, action.reason, positions_before
            )

        if action.kind == ActionKind.PARTIAL_CLOSE:
            return await self._partial_close(
                action.symbol, action.qty, action.reason, positions_before
            )

        if action.kind in (ActionKind.OPEN_LONG, ActionKind.OPEN_SHORT):
            return await self._open_position(action, equity)

        return True

    async def _open_position(self, action: TradeAction, equity: float) -> bool:
        """Open a new position."""
        bars = self._bars_cache.get(action.symbol)
        if bars is None:
            return False

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
                return False
            if result.filled_size <= 0:
                logger.error(
                    "Open %s %s reported no fill: %s", side, action.symbol, result
                )
                return False
            action.qty = min(float(result.filled_size), action.qty)
            size_usd = action.qty * current_price
            if result.avg_price > 0:
                current_price = result.avg_price
                size_usd = action.qty * current_price

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
        return True

    def _restore_position(self, pos: PositionState) -> None:
        """Restore a pre-execution position snapshot after failed reduce orders."""
        self._state.positions[pos.symbol] = PositionState(
            symbol=pos.symbol,
            qty=pos.qty,
            entry_price=pos.entry_price,
            entry_bar=pos.entry_bar,
            best_price=pos.best_price,
            partial_taken=pos.partial_taken,
        )

    async def _close_position(
        self,
        symbol: str,
        reason: str,
        positions_before: dict[str, PositionState] | None = None,
    ) -> bool:
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
            return True

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

        fill_qty = abs(pos.qty)
        exit_price = (
            float(self._bars_cache.get(symbol, pl.DataFrame())["close"][-1])
            if symbol in self._bars_cache
            else pos.entry_price
        )
        if not self._dry_run:
            result = await self._exchange.place_market_order(
                symbol, is_buy, size, reduce_only=True
            )
            if not result.success:
                logger.error("Failed to close %s: %s", symbol, result.message)
                self._restore_position(pos)
                return False
            if result.filled_size <= 0:
                logger.error("Close %s reported no fill: %s", symbol, result)
                self._restore_position(pos)
                return False
            fill_qty = min(float(result.filled_size), abs(pos.qty))
            if result.avg_price > 0:
                exit_price = result.avg_price

        remaining_qty = abs(pos.qty) - fill_qty
        if remaining_qty <= 1e-12:
            self._state.positions.pop(symbol, None)
        else:
            self._state.positions[symbol] = PositionState(
                symbol=symbol,
                qty=remaining_qty if pos.qty > 0 else -remaining_qty,
                entry_price=pos.entry_price,
                entry_bar=pos.entry_bar,
                best_price=pos.best_price,
                partial_taken=pos.partial_taken,
            )
        held = self._state.bar_count - pos.entry_bar

        # Compute PnL
        if pos.qty > 0:
            pnl = (exit_price - pos.entry_price) * fill_qty
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
        else:
            pnl = (pos.entry_price - exit_price) * fill_qty
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price * 100

        await self._notify.send(
            f"📉 CLOSE {side.upper()} {symbol} | held={held} bars | "
            f"pnl=${pnl:.2f} ({pnl_pct:+.1f}%) | {reason}"
        )
        self._journal.record_trade(
            bar=self._state.bar_count,
            kind="close",
            symbol=symbol,
            qty=fill_qty,
            price=exit_price,
            reason=reason,
            side=side,
            entry_price=pos.entry_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            held_bars=held,
        )
        return True

    async def _partial_close(
        self,
        symbol: str,
        qty: float,
        reason: str,
        positions_before: dict[str, PositionState] | None = None,
    ) -> bool:
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
            return True

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

        fill_qty = qty
        exit_price = (
            float(self._bars_cache.get(symbol, pl.DataFrame())["close"][-1])
            if symbol in self._bars_cache
            else pos_before.entry_price
        )
        if not self._dry_run:
            result = await self._exchange.place_market_order(
                symbol, is_buy, size, reduce_only=True
            )
            if not result.success:
                logger.error("Failed to partial close %s: %s", symbol, result.message)
                self._restore_position(pos_before)
                return False
            if result.filled_size <= 0:
                logger.error("Partial close %s reported no fill: %s", symbol, result)
                self._restore_position(pos_before)
                return False
            fill_qty = min(float(result.filled_size), qty)
            if result.avg_price > 0:
                exit_price = result.avg_price

        # tick() already adjusted pos.qty optimistically; correct it if the
        # reduce-only IOC filled less than requested.
        current = self._state.positions.get(symbol)
        if current is not None and fill_qty < qty:
            shortfall = qty - fill_qty
            current.qty += shortfall if pos_before.qty > 0 else -shortfall

        await self._notify.send(
            f"📊 PARTIAL {side.upper()} {symbol} | closed {fill_qty:.6f} | {reason}"
        )
        if pos_before.qty > 0:
            pnl = (exit_price - pos_before.entry_price) * fill_qty
        else:
            pnl = (pos_before.entry_price - exit_price) * fill_qty
        self._journal.record_trade(
            bar=self._state.bar_count,
            kind="partial_close",
            symbol=symbol,
            qty=fill_qty,
            price=exit_price,
            reason=reason,
            side=side,
            entry_price=pos_before.entry_price,
            pnl=pnl,
            pnl_pct=(pnl / (pos_before.entry_price * fill_qty) * 100)
            if pos_before.entry_price > 0
            else 0,
            held_bars=self._state.bar_count - pos_before.entry_bar,
        )
        return True

    async def _flatten_all(
        self, positions_before: dict[str, PositionState] | None = None
    ) -> bool:
        """Emergency: close all positions."""
        logger.warning("FLATTENING ALL POSITIONS")
        await self._notify.send("🚨 MAX DRAWDOWN — FLATTENING ALL POSITIONS")
        # Use pre-tick snapshot: tick() already deleted positions from state
        source = (
            positions_before if positions_before is not None else self._state.positions
        )
        symbols = list(source.keys())
        ok = True
        for symbol in symbols:
            ok = (
                await self._close_position(symbol, "flatten_all", positions_before)
                and ok
            )
        if not self._dry_run:
            await self._exchange.cancel_all_orders()
        return ok
