"""Core data types for cta-forge."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class Side(StrEnum):
    LONG = "long"
    SHORT = "short"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"


class TimeFrame(StrEnum):
    H1 = "1h"
    H4 = "4h"
    H6 = "6h"
    H8 = "8h"
    D1 = "1d"


@dataclass(frozen=True, slots=True)
class Bar:
    """Single OHLCV bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class Signal:
    """Alpha signal for a symbol."""

    symbol: str
    factor: str
    value: float  # [-1.0, +1.0]
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class Order:
    """Order to submit."""

    symbol: str
    side: Side
    size: float
    order_type: OrderType = OrderType.MARKET
    price: float | None = None


@dataclass(frozen=True, slots=True)
class Fill:
    """Executed fill."""

    symbol: str
    side: Side
    size: float
    price: float
    fee: float
    timestamp: datetime


@dataclass(slots=True)
class Position:
    """Current position state."""

    symbol: str
    size: float  # positive = long, negative = short
    entry_price: float
    unrealized_pnl: float = 0.0
    trailing_stop: float | None = None


@dataclass(slots=True)
class PortfolioState:
    """Snapshot of portfolio."""

    equity: float
    positions: dict[str, Position] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
