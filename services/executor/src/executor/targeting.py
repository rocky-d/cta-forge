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
    ) -> None:
        delta_notional = delta * equity
        if abs(delta_notional) < min_notional:
            return
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

    for symbol in symbols:
        price = prices.get(symbol)
        if price is None or price <= 0:
            continue
        current = cur.get(symbol, 0.0)
        target = float(target_weights.get(symbol, 0.0))

        if abs(current) > EPS and current * target < 0:
            append_order(
                symbol,
                price,
                current,
                target,
                -current,
                reduce_only=True,
            )
            append_order(
                symbol,
                price,
                0.0,
                target,
                target,
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
