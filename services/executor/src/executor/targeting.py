"""Target-weight utilities for portfolio-style strategy execution.

Weights are signed fractions of account equity. Positive means long exposure,
negative means short exposure. This module intentionally contains no exchange
I/O so the same target construction can be used by research backtests and live
execution adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Mapping, Protocol

EPS = 1e-12


@dataclass(frozen=True)
class StrategyProfile:
    """Stable metadata for one selectable strategy profile."""

    slug: str
    name: str
    description: str = ""
    timeframe_hours: int = 6


@dataclass(frozen=True)
class PortfolioTarget:
    """Desired signed portfolio weights for one timestamp."""

    timestamp: datetime
    weights: Mapping[str, float]
    gross_cap: float = 1.0

    @property
    def gross(self) -> float:
        """Absolute gross exposure requested by the target."""
        return sum(abs(float(weight)) for weight in self.weights.values())

    def capped(self) -> PortfolioTarget:
        """Return a copy scaled to ``gross_cap`` when needed."""
        return PortfolioTarget(
            timestamp=self.timestamp,
            weights=normalize_gross(self.weights, gross_cap=self.gross_cap),
            gross_cap=self.gross_cap,
        )


@dataclass(frozen=True)
class SleeveTarget:
    """Target weights emitted by one strategy sleeve."""

    name: str
    weights: Mapping[str, float]
    allocation: float = 1.0


class TargetWeightStrategy(Protocol):
    """Strategy boundary for target-weight portfolio strategies."""

    profile: StrategyProfile

    def target(self, timestamp: datetime) -> PortfolioTarget:
        """Return target weights known at ``timestamp``."""
        ...


@dataclass(frozen=True)
class TargetOrder:
    """Delta needed to move one symbol toward a target weight."""

    symbol: str
    side: Literal["buy", "sell"]
    qty: float
    current_weight: float
    target_weight: float
    delta_weight: float
    delta_notional: float
    reduce_only: bool


def normalize_gross(
    weights: Mapping[str, float], gross_cap: float = 1.0
) -> dict[str, float]:
    """Scale weights down when absolute gross exposure exceeds ``gross_cap``."""
    clean = {
        symbol: float(weight) for symbol, weight in weights.items() if abs(weight) > EPS
    }
    gross = sum(abs(weight) for weight in clean.values())
    if gross <= gross_cap or gross <= EPS:
        return clean
    scale = gross_cap / gross
    return {symbol: weight * scale for symbol, weight in clean.items()}


def combine_sleeves(
    sleeves: list[SleeveTarget],
    *,
    gate_scale: float = 1.0,
    gross_cap: float = 1.0,
) -> dict[str, float]:
    """Combine multiple sleeve targets, apply a scalar gate, then cap gross."""
    combined: dict[str, float] = {}
    for sleeve in sleeves:
        for symbol, weight in sleeve.weights.items():
            combined[symbol] = combined.get(symbol, 0.0) + sleeve.allocation * float(
                weight
            )
    gated = {symbol: weight * gate_scale for symbol, weight in combined.items()}
    return normalize_gross(gated, gross_cap=gross_cap)


def current_weights(
    positions: Mapping[str, float],
    prices: Mapping[str, float],
    equity: float,
) -> dict[str, float]:
    """Convert signed base-unit positions into signed equity weights."""
    if equity <= EPS:
        return {}
    weights: dict[str, float] = {}
    for symbol, qty in positions.items():
        price = prices.get(symbol)
        if price is None or price <= 0:
            continue
        weights[symbol] = float(qty) * float(price) / equity
    return weights


def weights_to_orders(
    positions: Mapping[str, float],
    prices: Mapping[str, float],
    equity: float,
    target_weights: Mapping[str, float],
    *,
    min_notional: float = 10.0,
    max_notional: float | None = None,
) -> list[TargetOrder]:
    """Create delta orders to move current positions toward target weights.

    The function does not mutate state and does not submit orders. Returned
    orders are sorted reduce-first so an execution adapter can lower exposure
    before increasing it.
    """
    if equity <= EPS:
        return []

    cur = current_weights(positions, prices, equity)
    symbols = sorted(set(cur) | set(target_weights))
    orders: list[TargetOrder] = []

    def append_order(
        symbol: str,
        price: float,
        current: float,
        target: float,
        delta: float,
        *,
        reduce_only: bool,
    ) -> bool:
        delta_notional = delta * equity
        if not reduce_only and max_notional is not None and max_notional > 0:
            cap = float(max_notional)
            delta_notional = max(-cap, min(cap, delta_notional))
            delta = delta_notional / equity
        if abs(delta_notional) < min_notional:
            return False
        side: Literal["buy", "sell"] = "buy" if delta_notional > 0 else "sell"
        orders.append(
            TargetOrder(
                symbol=symbol,
                side=side,
                qty=abs(delta_notional) / price,
                current_weight=current,
                target_weight=target,
                delta_weight=delta,
                delta_notional=delta_notional,
                reduce_only=reduce_only,
            )
        )
        return True

    for symbol in symbols:
        price = prices.get(symbol)
        if price is None or price <= 0:
            continue
        current = cur.get(symbol, 0.0)
        target = float(target_weights.get(symbol, 0.0))

        if abs(current) > EPS and current * target < 0:
            close_delta = -current
            open_delta = target
            full_delta = target - current
            close_notional = abs(close_delta * equity)
            open_notional = abs(open_delta * equity)
            full_notional = abs(full_delta * equity)
            close_tradable = close_notional >= min_notional
            open_tradable = open_notional >= min_notional
            split_would_skip_leg = not close_tradable or not open_tradable
            full_exceeds_cap = (
                max_notional is not None
                and max_notional > 0
                and full_notional > float(max_notional)
            )

            # A sign flip is usually split into reduce-only close then open, so
            # large flips de-risk before adding opposite exposure.  However,
            # Hyperliquid's per-order minimum can make one split leg invalid
            # even when the net crossing order is valid.  In that case, send a
            # single non-reduce crossing order; it nets against the current
            # position and avoids permanent sub-minimum dust blocking flips.
            if (
                split_would_skip_leg
                and full_notional >= min_notional
                and (not full_exceeds_cap or not close_tradable)
            ):
                append_order(
                    symbol,
                    price,
                    current,
                    target,
                    full_delta,
                    reduce_only=False,
                )
                continue

            if close_tradable:
                append_order(
                    symbol,
                    price,
                    current,
                    target,
                    close_delta,
                    reduce_only=True,
                )
            if open_tradable:
                append_order(
                    symbol,
                    price,
                    0.0,
                    target,
                    open_delta,
                    reduce_only=False,
                )
            continue

        delta = target - current
        reduce_only = abs(current) > EPS and abs(target) < abs(current)
        append_order(
            symbol,
            price,
            current,
            target,
            delta,
            reduce_only=reduce_only,
        )

    return sorted(orders, key=lambda order: (not order.reduce_only, order.symbol))
