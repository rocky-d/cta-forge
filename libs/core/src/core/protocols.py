"""Protocol definitions for inter-service contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import polars as pl


class DataProvider(Protocol):
    """Contract for data-service client."""

    def get_symbols(self) -> list[str]: ...
    def get_bars(self, symbol: str, tf: str, start: str, end: str) -> pl.DataFrame: ...
    def sync(self) -> None: ...


class AlphaFactor(Protocol):
    """Contract for a single alpha factor."""

    @property
    def name(self) -> str: ...

    def compute(self, bars: pl.DataFrame) -> pl.DataFrame:
        """Compute factor signal. Returns DataFrame with 'timestamp' and 'signal' columns.

        Signal value domain: [-1.0, +1.0].
        """
        ...


class SignalComposer(Protocol):
    """Contract for multi-factor signal composition."""

    def compose(
        self,
        signals: dict[str, pl.DataFrame],
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Compose multiple factor signals into a single signal per symbol.

        Returns: {symbol: composite_signal}.
        """
        ...


class AssetSelector(Protocol):
    """Contract for dynamic asset selection."""

    def select(self, universe: dict[str, pl.DataFrame]) -> list[str]:
        """Select tradable symbols from the full universe."""
        ...


class Allocator(Protocol):
    """Contract for position allocation."""

    def allocate(self, signals: dict[str, float], equity: float) -> dict[str, float]:
        """Allocate capital based on signals.

        Returns: {symbol: target_notional}. Positive = long, negative = short.
        """
        ...


class RiskManager(Protocol):
    """Contract for risk management."""

    def check(
        self,
        positions: dict[str, float],
        bars: dict[str, pl.DataFrame],
    ) -> dict[str, float]:
        """Adjust positions based on risk rules.

        Returns: adjusted {symbol: target_notional}.
        """
        ...


class ExecutionEngine(Protocol):
    """Contract for execution layer (backtest or live)."""

    async def get_equity(self) -> float: ...
    async def get_positions(self) -> dict[str, float]: ...
    async def submit_target(self, target: dict[str, float]) -> None: ...
