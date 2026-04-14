"""Protocol definitions for inter-service contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import polars as pl


class AlphaFactor(Protocol):
    """Contract for a single alpha factor."""

    @property
    def name(self) -> str: ...

    def compute(self, bars: pl.DataFrame) -> pl.DataFrame:
        """Compute factor signal. Returns DataFrame with 'timestamp' and 'signal' columns.

        Signal value domain: [-1.0, +1.0].
        """
        ...
