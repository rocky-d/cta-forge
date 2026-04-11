"""Exchange adapter protocol — interface for exchange operations.

Design: Protocol (structural subtyping) for duck-typing + static type safety.
Engine only talks to this interface, never directly to exchange APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from decimal import Decimal


@dataclass(frozen=True)
class OrderResult:
    """Result of placing or canceling an order."""

    order_id: str
    success: bool
    message: str = ""
    avg_price: float = 0.0
    filled_size: float = 0.0


@dataclass(frozen=True)
class Position:
    """Current position for a symbol."""

    symbol: str
    size: Decimal  # positive=long, negative=short, zero=flat
    entry_price: Decimal
    unrealized_pnl: Decimal
    leverage: int


@dataclass(frozen=True)
class AccountState:
    """Account-level state."""

    equity: Decimal
    available_balance: Decimal
    total_margin_used: Decimal
    positions: list[Position]


@dataclass(frozen=True)
class MarketSnapshot:
    """Current market data snapshot."""

    symbol: str
    mid_price: Decimal
    best_bid: Decimal
    best_ask: Decimal
    mark_price: Decimal
    funding_rate: Decimal
    timestamp_ms: int


@runtime_checkable
class ExchangeAdapter(Protocol):
    """Protocol interface for exchange operations.

    Implementations handle the actual REST/WS API calls.
    The trading loop only interacts through this interface.
    """

    async def get_account_state(self) -> AccountState:
        """Get full account state including positions."""
        ...

    async def get_position(self, symbol: str) -> Position | None:
        """Get position for a specific symbol, None if flat."""
        ...

    async def get_market_snapshot(self, symbol: str) -> MarketSnapshot:
        """Get current market data snapshot."""
        ...

    async def place_market_order(
        self,
        symbol: str,
        is_buy: bool,
        size: Decimal,
        *,
        reduce_only: bool = False,
    ) -> OrderResult:
        """Place a market order (IOC at aggressive price)."""
        ...

    async def place_limit_order(
        self,
        symbol: str,
        is_buy: bool,
        size: Decimal,
        price: Decimal,
        *,
        reduce_only: bool = False,
        post_only: bool = False,
    ) -> OrderResult:
        """Place a limit order."""
        ...

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order by ID. Returns True if successful."""
        ...

    async def cancel_all_orders(self, symbol: str | None = None) -> int:
        """Cancel all open orders, optionally filtered by symbol."""
        ...

    async def set_leverage(self, symbol: str, leverage: int, cross: bool = True) -> bool:
        """Set leverage for a symbol."""
        ...

    async def transfer_to_perp(self, amount: Decimal) -> bool:
        """Transfer USDC from spot to perp account."""
        ...

    async def close(self) -> None:
        """Clean up connections."""
        ...
