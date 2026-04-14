"""Parquet-based local data store for kline data."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import polars as pl
from core.constants import PARQUET_COMPRESSION

logger = logging.getLogger(__name__)


class ParquetStore:
    """Manages parquet files for kline data.

    Layout: {data_dir}/{symbol}/{interval}.parquet
    """

    def __init__(self, data_dir: str | Path) -> None:
        self._root = Path(data_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, interval: str) -> Path:
        d = self._root / symbol.upper()
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{interval}.parquet"

    def has_data(self, symbol: str, interval: str) -> bool:
        return self._path(symbol, interval).exists()

    def read(
        self,
        symbol: str,
        interval: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        """Read bars from parquet, optionally filtered by time range."""
        path = self._path(symbol, interval)
        if not path.exists():
            return pl.DataFrame()

        df = pl.read_parquet(path)
        if df.is_empty():
            return df

        if start is not None:
            df = df.filter(pl.col("open_time") >= start)
        if end is not None:
            df = df.filter(pl.col("open_time") <= end)

        return df.sort("open_time")

    def write(self, symbol: str, interval: str, df: pl.DataFrame) -> int:
        """Write bars to parquet, merging with existing data.

        Returns the total number of bars after merge.
        """
        if df.is_empty():
            return 0

        path = self._path(symbol, interval)

        if path.exists():
            existing = pl.read_parquet(path)
            df = (
                pl.concat([existing, df]).unique(subset=["open_time"]).sort("open_time")
            )

        df.write_parquet(path, compression=PARQUET_COMPRESSION)
        logger.info("Wrote %d bars for %s/%s", len(df), symbol, interval)
        return len(df)

    def latest_timestamp(self, symbol: str, interval: str) -> datetime | None:
        """Get the latest open_time for a symbol/interval pair."""
        path = self._path(symbol, interval)
        if not path.exists():
            return None

        df = pl.read_parquet(path, columns=["open_time"])
        if df.is_empty():
            return None

        ts = df["open_time"].max()
        if not isinstance(ts, datetime):
            return None
        return ts

    def symbols(self) -> list[str]:
        """List all symbols that have data stored."""
        if not self._root.exists():
            return []
        return sorted(d.name for d in self._root.iterdir() if d.is_dir())

    def coverage(self, symbol: str, interval: str) -> dict[str, str | int | None]:
        """Get data coverage info for a symbol/interval."""
        path = self._path(symbol, interval)
        if not path.exists():
            return {
                "symbol": symbol,
                "interval": interval,
                "bars": 0,
                "start": None,
                "end": None,
            }

        df = pl.read_parquet(path, columns=["open_time"])
        if df.is_empty():
            return {
                "symbol": symbol,
                "interval": interval,
                "bars": 0,
                "start": None,
                "end": None,
            }

        return {
            "symbol": symbol,
            "interval": interval,
            "bars": len(df),
            "start": str(df["open_time"].min()),
            "end": str(df["open_time"].max()),
        }
