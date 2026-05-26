"""Target-weight execution helpers for the live engine.

This module keeps portfolio-target reconciliation separate from the main
``LiveEngine`` loop. It owns no strategy logic: target construction remains in
``TargetWeightStrategy`` implementations, while this module only normalizes
symbols, derives market orders, records diagnostics, and applies successful
fills to engine state.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from .decision import EngineState, PositionState
from .journal import LiveJournalStore
from .targeting import (
    PortfolioTarget,
    TargetOrder,
    TargetWeightStrategy,
    weights_to_orders,
)

if TYPE_CHECKING:
    from exchange.adapter import AccountState, ExchangeAdapter

logger = logging.getLogger(__name__)


def normalize_live_symbol(symbol: str) -> str:
    """Convert research symbols like BTCUSDT to live symbols like BTC."""
    return symbol[:-4] if symbol.endswith("USDT") else symbol


def normalize_target_weights(
    weights: dict[str, float], allowed_symbols: set[str]
) -> tuple[dict[str, float], dict[str, float]]:
    """Normalize target symbols and keep rejected targets for diagnostics."""
    normalized: dict[str, float] = {}
    ignored: dict[str, float] = {}
    for raw_symbol, weight in weights.items():
        live_symbol = normalize_live_symbol(raw_symbol)
        weight = float(weight)
        if live_symbol not in allowed_symbols:
            logger.warning(
                "Ignoring target for %s outside configured universe", raw_symbol
            )
            ignored[raw_symbol] = ignored.get(raw_symbol, 0.0) + weight
            continue
        normalized[live_symbol] = normalized.get(live_symbol, 0.0) + weight
    return normalized, ignored


def sync_target_state_from_account(
    state: EngineState, account: "AccountState", symbols: set[str] | None = None
) -> None:
    """Use managed exchange positions as source of truth before target orders."""
    next_positions: dict[str, PositionState] = {}
    for pos in account.positions:
        if symbols is not None and pos.symbol not in symbols:
            continue
        size = float(pos.size)
        if abs(size) <= 1e-12:
            continue
        existing = state.positions.get(pos.symbol)
        next_positions[pos.symbol] = PositionState(
            symbol=pos.symbol,
            qty=size,
            entry_price=float(pos.entry_price),
            entry_bar=existing.entry_bar if existing else state.bar_count,
            best_price=existing.best_price if existing else float(pos.entry_price),
            partial_taken=existing.partial_taken if existing else False,
        )
    state.positions = next_positions


async def fetch_target_prices(
    exchange: "ExchangeAdapter", symbols: set[str]
) -> dict[str, float]:
    """Fetch live mark prices for target reconciliation."""
    prices: dict[str, float] = {}
    for symbol in sorted(symbols):
        try:
            snap = await exchange.get_market_snapshot(symbol)
        except Exception:
            logger.exception("Failed to fetch target price for %s", symbol)
            continue
        price = float(snap.mark_price or snap.mid_price)
        if price > 0:
            prices[symbol] = price
    return prices


def record_target_diagnostics(
    journal: LiveJournalStore,
    *,
    state: EngineState,
    profile: str,
    now: datetime,
    target: PortfolioTarget,
    target_weights: dict[str, float],
    orders: list[TargetOrder],
    ignored_weights: dict[str, float] | None = None,
) -> None:
    """Persist target portfolio diagnostics for shadow validation."""
    normalized_gross = sum(abs(weight) for weight in target_weights.values())
    staleness = (now - target.timestamp).total_seconds()
    journal.record_target(
        bar=state.bar_count + 1,
        profile=profile,
        target_ts=target.timestamp.isoformat(),
        staleness_seconds=staleness,
        target_gross=target.gross,
        normalized_gross=normalized_gross,
        weights=target_weights,
        orders=[
            {
                "symbol": order.symbol,
                "side": order.side,
                "qty": float(order.qty),
                "current_weight": float(order.current_weight),
                "target_weight": float(order.target_weight),
                "delta_weight": float(order.delta_weight),
                "delta_notional": float(order.delta_notional),
                "reduce_only": order.reduce_only,
            }
            for order in orders
        ],
        ignored_weights=ignored_weights,
    )


async def execute_target_order(
    *,
    exchange: "ExchangeAdapter",
    journal: LiveJournalStore,
    state: EngineState,
    profile: str,
    dry_run: bool,
    order: TargetOrder,
    price: float,
) -> bool:
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
    fill_qty = order.qty
    if dry_run:
        logger.info(
            "[DRY RUN] Would target-order %s %s %.6f @ ~%.2f reduce_only=%s",
            order.side,
            order.symbol,
            order.qty,
            price,
            order.reduce_only,
        )
    else:
        result = await exchange.place_market_order(
            order.symbol,
            is_buy,
            size,
            reduce_only=order.reduce_only,
        )
        if not result.success:
            logger.error("Failed target order %s: %s", order.symbol, result.message)
            return False
        if result.filled_size <= 0:
            logger.error("Target order %s reported no fill: %s", order.symbol, result)
            return False
        fill_qty = float(result.filled_size)
        if result.avg_price > 0:
            fill_price = result.avg_price

    filled_order = TargetOrder(
        symbol=order.symbol,
        side=order.side,
        qty=fill_qty,
        current_weight=order.current_weight,
        target_weight=order.target_weight,
        delta_weight=order.delta_weight,
        delta_notional=(fill_qty * fill_price) * (1 if is_buy else -1),
        reduce_only=order.reduce_only,
    )
    apply_target_fill(state, filled_order, fill_price)
    journal.record_trade(
        bar=state.bar_count,
        kind="target_buy" if is_buy else "target_sell",
        symbol=order.symbol,
        qty=fill_qty,
        price=fill_price,
        reason=f"target:{profile}",
        side="long" if is_buy else "short",
    )
    return True


def apply_target_fill(state: EngineState, order: TargetOrder, price: float) -> None:
    """Apply a successfully executed target order to local engine state."""
    signed_qty = order.qty if order.side == "buy" else -order.qty
    current = state.positions.get(order.symbol)
    old_qty = current.qty if current is not None else 0.0
    new_qty = old_qty + signed_qty

    if abs(new_qty) <= 1e-12 or (order.reduce_only and old_qty * new_qty <= 0):
        state.positions.pop(order.symbol, None)
        return

    if current is None or old_qty * new_qty <= 0:
        entry_price = price
        entry_bar = state.bar_count
        best_price = price
        partial_taken = False
    elif abs(new_qty) > abs(old_qty):
        added_qty = abs(signed_qty)
        entry_price = (abs(old_qty) * current.entry_price + added_qty * price) / abs(
            new_qty
        )
        entry_bar = current.entry_bar
        best_price = current.best_price
        partial_taken = current.partial_taken
    else:
        entry_price = current.entry_price
        entry_bar = current.entry_bar
        best_price = current.best_price
        partial_taken = current.partial_taken

    state.positions[order.symbol] = PositionState(
        symbol=order.symbol,
        qty=new_qty,
        entry_price=entry_price,
        entry_bar=entry_bar,
        best_price=best_price,
        partial_taken=partial_taken,
    )


async def execute_target_portfolio(
    *,
    exchange: "ExchangeAdapter",
    journal: LiveJournalStore,
    state: EngineState,
    account: "AccountState",
    equity: float,
    target_strategy: TargetWeightStrategy | None,
    symbols: list[str],
    profile: str,
    dry_run: bool,
    min_order_notional: float,
    max_order_notional: float | None = None,
) -> list[TargetOrder]:
    """Reconcile a target-weight strategy into market-order deltas."""
    if target_strategy is None:
        return []

    managed = set(symbols)
    sync_target_state_from_account(state, account, managed)
    now = datetime.now(tz=UTC)
    target = target_strategy.target(now).capped()
    target_weights, ignored_weights = normalize_target_weights(
        dict(target.weights), set(symbols)
    )
    unmanaged_positions = [
        pos.symbol for pos in account.positions if pos.symbol not in managed
    ]
    if unmanaged_positions:
        logger.warning(
            "Ignoring unmanaged account position(s) outside LIVE_SYMBOLS: %s",
            ",".join(sorted(unmanaged_positions)),
        )
    positions = {
        pos.symbol: float(pos.size)
        for pos in account.positions
        if pos.symbol in managed
    }
    order_symbols = set(positions) | set(target_weights)
    prices = await fetch_target_prices(exchange, order_symbols)
    orders = weights_to_orders(
        positions,
        prices,
        equity,
        target_weights,
        min_notional=min_order_notional,
        max_notional=max_order_notional,
    )
    record_target_diagnostics(
        journal,
        state=state,
        profile=profile,
        now=now,
        target=target,
        target_weights=target_weights,
        orders=orders,
        ignored_weights=ignored_weights,
    )

    for order in orders:
        price = prices.get(order.symbol)
        if price is None:
            continue
        ok = await execute_target_order(
            exchange=exchange,
            journal=journal,
            state=state,
            profile=profile,
            dry_run=dry_run,
            order=order,
            price=price,
        )
        if not ok:
            logger.error("Stopping target order batch after failed %s", order.symbol)
            break

    logger.info(
        "Target profile %s produced %d order(s), gross=%.3f",
        profile,
        len(orders),
        target.gross,
    )
    return orders
