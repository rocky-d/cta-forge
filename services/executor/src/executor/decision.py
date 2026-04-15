"""V10G decision engine: unified trading logic for live and backtest.

This module is the single source of truth for all v10g trade decisions:
opening, closing, partial take-profit, trailing stops, position sizing,
vol scaling, DD breaker, max hold, stop tightening, and signal reversal.

Both the live engine and the backtest script delegate to this module,
eliminating logic divergence between paper and production.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

logger = logging.getLogger(__name__)


# ── Parameters ───────────────────────────────────────────────────


@dataclass
class V10GStrategyParams:
    """Complete parameter set for v10g strategy.

    Source of truth: backtest v10g_maxrange.py best-performing config.
    """

    signal_threshold: float = 0.40
    min_hold_bars: int = 16
    atr_stop_mult: float = 5.0
    risk_per_trade: float = 0.015
    max_positions: int = 5
    rebalance_every: int = 4
    partial_take_profit: float = 2.5
    target_vol: float = 0.12
    max_hold_bars: int = 100
    tighten_stop_after_atr: float = 2.0
    tightened_stop_mult: float = 3.0
    dd_circuit_breaker: float = 0.08
    max_drawdown: float = 0.15
    risk_parity: bool = True
    signal_reversal_threshold: float = 0.15
    max_single_position_pct: float = 0.15
    commission: float = 0.0004


# ── Position / state types ───────────────────────────────────────


class ActionKind(StrEnum):
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE = "close"
    PARTIAL_CLOSE = "partial_close"
    FLATTEN_ALL = "flatten_all"


@dataclass
class TradeAction:
    """A single trade decision emitted by the decision engine."""

    kind: ActionKind
    symbol: str
    qty: float = 0.0
    reason: str = ""


@dataclass
class PositionState:
    """Tracked position metadata (shared between live and backtest)."""

    symbol: str
    qty: float  # positive = long, negative = short
    entry_price: float
    entry_bar: int
    best_price: float  # highest for long, lowest for short
    partial_taken: bool = False


@dataclass
class EngineState:
    """Mutable state carried across ticks."""

    positions: dict[str, PositionState] = field(default_factory=dict)
    bar_count: int = 0
    initial_equity: float = 0.0
    peak_equity: float = 0.0
    recent_returns: list[float] = field(default_factory=list)


# ── Market snapshot (per bar) ────────────────────────────────────


@dataclass
class BarSnapshot:
    """Per-symbol market data for the current bar."""

    close: float
    atr: float
    rvol: float  # realized vol (std of returns over lookback)
    signal: float


# ── Decision engine ──────────────────────────────────────────────


class V10GDecisionEngine:
    """Stateless decision engine: given state + market data, emit actions.

    This class contains zero execution logic (no exchange calls, no I/O).
    Callers (live engine or backtest loop) handle execution.
    """

    def __init__(self, params: V10GStrategyParams | None = None) -> None:
        self.p = params or V10GStrategyParams()

    def tick(
        self,
        state: EngineState,
        equity: float,
        snapshots: dict[str, BarSnapshot],
    ) -> list[TradeAction]:
        """Process one bar and return a list of trade actions.

        Args:
            state: mutable engine state (positions, counters, etc.)
            equity: current total equity (mark-to-market)
            snapshots: per-symbol market data for this bar

        Returns:
            Ordered list of actions to execute.
        """
        actions: list[TradeAction] = []

        state.bar_count += 1
        state.peak_equity = max(state.peak_equity, equity)

        # ── Drawdown calculations ────────────────────────────────
        cur_dd = (
            (state.peak_equity - equity) / state.peak_equity
            if state.peak_equity > 0
            else 0.0
        )

        # Hard stop: flatten everything
        if cur_dd >= self.p.max_drawdown:
            logger.warning(
                "MAX DRAWDOWN %.1f%% >= %.1f%% — flatten all",
                cur_dd * 100,
                self.p.max_drawdown * 100,
            )
            for sym in list(state.positions):
                actions.append(
                    TradeAction(
                        kind=ActionKind.FLATTEN_ALL,
                        symbol=sym,
                        reason=f"max_drawdown_{cur_dd:.2%}",
                    )
                )
            return actions

        # Vol scaling
        vol_scale = self._vol_scale(state)
        # DD breaker scaling
        dd_scale = (
            0.5
            if self.p.dd_circuit_breaker > 0 and cur_dd > self.p.dd_circuit_breaker
            else 1.0
        )

        # ── Update recent returns (for vol scaling) ──────────────
        # Caller should have updated equity before calling tick.
        # We track returns here if we have prior equity.
        if len(state.recent_returns) > 120:
            state.recent_returns.pop(0)

        # ── Close decisions ──────────────────────────────────────
        close_actions = self._close_decisions(state, snapshots)
        actions.extend(close_actions)

        # Apply closes to state immediately (so rebalance sees updated slots)
        for a in close_actions:
            if a.symbol in state.positions:
                del state.positions[a.symbol]

        # ── Partial take-profit ──────────────────────────────────
        partial_actions = self._partial_tp_decisions(state, snapshots)
        actions.extend(partial_actions)

        # Apply partials to state
        for a in partial_actions:
            pos = state.positions.get(a.symbol)
            if pos is not None:
                pos.qty -= a.qty if pos.qty > 0 else -a.qty
                pos.partial_taken = True

        # ── Open decisions (only on rebalance bars) ──────────────
        if state.bar_count % self.p.rebalance_every == 0 or state.bar_count == 1:
            open_actions = self._open_decisions(
                state, equity, snapshots, vol_scale, dd_scale
            )
            actions.extend(open_actions)

        # ── Update best_price for remaining positions ────────────
        for sym, pos in state.positions.items():
            snap = snapshots.get(sym)
            if snap is None:
                continue
            if pos.qty > 0:
                pos.best_price = max(pos.best_price, snap.close)
            else:
                pos.best_price = min(pos.best_price, snap.close)

        return actions

    # ── Close logic ──────────────────────────────────────────────

    def _close_decisions(
        self,
        state: EngineState,
        snapshots: dict[str, BarSnapshot],
    ) -> list[TradeAction]:
        actions: list[TradeAction] = []

        for sym, pos in list(state.positions.items()):
            snap = snapshots.get(sym)
            if snap is None:
                actions.append(
                    TradeAction(kind=ActionKind.CLOSE, symbol=sym, reason="no_data")
                )
                continue

            held = state.bar_count - pos.entry_bar
            cp = snap.close
            atr = snap.atr

            # 1. Max hold bars
            if self.p.max_hold_bars > 0 and held >= self.p.max_hold_bars:
                actions.append(
                    TradeAction(
                        kind=ActionKind.CLOSE,
                        symbol=sym,
                        reason=f"max_hold_{held}",
                    )
                )
                continue

            # 2. Trailing stop (with tightening)
            stop_mult = self.p.atr_stop_mult
            if self.p.tighten_stop_after_atr > 0 and atr > 0:
                unrealized_atr = (
                    (cp - pos.entry_price) / atr
                    if pos.qty > 0
                    else (pos.entry_price - cp) / atr
                )
                if unrealized_atr > self.p.tighten_stop_after_atr:
                    stop_mult = self.p.tightened_stop_mult

            if pos.qty > 0:
                if (
                    cp < pos.best_price - stop_mult * atr
                    and held >= self.p.min_hold_bars
                ):
                    actions.append(
                        TradeAction(
                            kind=ActionKind.CLOSE,
                            symbol=sym,
                            reason=f"trailing_stop_long_{stop_mult:.1f}x",
                        )
                    )
                    continue
            else:
                if (
                    cp > pos.best_price + stop_mult * atr
                    and held >= self.p.min_hold_bars
                ):
                    actions.append(
                        TradeAction(
                            kind=ActionKind.CLOSE,
                            symbol=sym,
                            reason=f"trailing_stop_short_{stop_mult:.1f}x",
                        )
                    )
                    continue

            # 3. Signal reversal
            if held >= self.p.min_hold_bars:
                sig = snap.signal
                if pos.qty > 0 and sig < -self.p.signal_reversal_threshold:
                    actions.append(
                        TradeAction(
                            kind=ActionKind.CLOSE,
                            symbol=sym,
                            reason=f"signal_reversal_{sig:.2f}",
                        )
                    )
                elif pos.qty < 0 and sig > self.p.signal_reversal_threshold:
                    actions.append(
                        TradeAction(
                            kind=ActionKind.CLOSE,
                            symbol=sym,
                            reason=f"signal_reversal_{sig:.2f}",
                        )
                    )

        return actions

    # ── Partial take-profit ──────────────────────────────────────

    def _partial_tp_decisions(
        self,
        state: EngineState,
        snapshots: dict[str, BarSnapshot],
    ) -> list[TradeAction]:
        if self.p.partial_take_profit <= 0:
            return []

        actions: list[TradeAction] = []
        for sym, pos in state.positions.items():
            if pos.partial_taken:
                continue
            snap = snapshots.get(sym)
            if snap is None or snap.atr <= 0:
                continue

            cp = snap.close
            tp_dist = self.p.partial_take_profit * snap.atr

            if pos.qty > 0 and cp > pos.entry_price + tp_dist:
                half_qty = pos.qty / 2
                actions.append(
                    TradeAction(
                        kind=ActionKind.PARTIAL_CLOSE,
                        symbol=sym,
                        qty=half_qty,
                        reason=f"partial_tp_long_{cp:.2f}",
                    )
                )
            elif pos.qty < 0 and cp < pos.entry_price - tp_dist:
                half_qty = abs(pos.qty) / 2
                actions.append(
                    TradeAction(
                        kind=ActionKind.PARTIAL_CLOSE,
                        symbol=sym,
                        qty=half_qty,
                        reason=f"partial_tp_short_{cp:.2f}",
                    )
                )

        return actions

    # ── Open logic ───────────────────────────────────────────────

    def _open_decisions(
        self,
        state: EngineState,
        equity: float,
        snapshots: dict[str, BarSnapshot],
        vol_scale: float,
        dd_scale: float,
    ) -> list[TradeAction]:
        available = self.p.max_positions - len(state.positions)
        if available <= 0:
            return []

        # Candidates: symbols not in positions with signal above threshold
        candidates = []
        for sym, snap in snapshots.items():
            if sym in state.positions:
                continue
            if abs(snap.signal) >= self.p.signal_threshold:
                candidates.append((sym, snap))

        # Sort by signal strength
        candidates.sort(key=lambda x: abs(x[1].signal), reverse=True)

        actions: list[TradeAction] = []
        for sym, snap in candidates[:available]:
            if snap.atr <= 0 or snap.close <= 0:
                continue

            qty = self._size_position(equity, snap, vol_scale, dd_scale)
            if qty * snap.close < 10:
                continue

            if snap.signal > 0:
                actions.append(
                    TradeAction(
                        kind=ActionKind.OPEN_LONG,
                        symbol=sym,
                        qty=qty,
                        reason=f"signal_{snap.signal:.2f}",
                    )
                )
            else:
                actions.append(
                    TradeAction(
                        kind=ActionKind.OPEN_SHORT,
                        symbol=sym,
                        qty=qty,
                        reason=f"signal_{snap.signal:.2f}",
                    )
                )

        return actions

    # ── Position sizing ──────────────────────────────────────────

    def _size_position(
        self,
        equity: float,
        snap: BarSnapshot,
        vol_scale: float,
        dd_scale: float,
    ) -> float:
        """Compute position size in base units."""
        if self.p.risk_parity:
            # Risk parity: size inversely proportional to realized vol
            rv = snap.rvol if snap.rvol > 0 else 0.01
            target_risk = equity * self.p.risk_per_trade / self.p.max_positions
            qty = min(
                target_risk / (rv * snap.close * np.sqrt(20)),
                equity * self.p.max_single_position_pct / snap.close,
            )
        else:
            # ATR-based sizing
            normalized_atr = snap.atr / snap.close
            mean_atr = 0.02
            iv = np.clip(mean_atr / normalized_atr, 0.5, 2.0)
            qty = min(
                equity
                * self.p.risk_per_trade
                * iv
                * vol_scale
                / (self.p.atr_stop_mult * snap.atr),
                equity * self.p.max_single_position_pct / snap.close,
            )

        qty *= vol_scale * dd_scale
        return abs(qty)

    # ── Vol scaling ──────────────────────────────────────────────

    def _vol_scale(self, state: EngineState) -> float:
        """Target-vol scaling based on recent portfolio returns."""
        if self.p.target_vol <= 0 or len(state.recent_returns) < 20:
            return 1.0
        window = state.recent_returns[-60:]
        rv = np.std(window) * np.sqrt(4 * 365)  # annualize 6h returns
        if rv <= 0:
            return 1.0
        return float(np.clip(self.p.target_vol / rv, 0.3, 2.0))
