"""Tests for ParquetStore."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest
from data_service.store import ParquetStore


@pytest.fixture()
def store(tmp_path):
    return ParquetStore(tmp_path / "data")


def _make_bars(n: int = 5, start_hour: int = 0) -> pl.DataFrame:
    """Create a sample bars DataFrame."""
    rows = []
    for i in range(n):
        h = start_hour + i * 6
        rows.append(
            {
                "open_time": datetime(2024, 1, 1, h % 24, 0, 0, tzinfo=UTC),
                "open": 100.0 + i,
                "high": 105.0 + i,
                "low": 95.0 + i,
                "close": 102.0 + i,
                "volume": 1000.0 + i * 100,
                "close_time": datetime(2024, 1, 1, (h + 6) % 24, 0, 0, tzinfo=UTC),
                "quote_volume": 100000.0,
                "trades": 500 + i,
                "taker_buy_volume": 500.0,
                "taker_buy_quote_volume": 50000.0,
            }
        )
    return pl.DataFrame(rows)


def test_write_and_read(store: ParquetStore):
    df = _make_bars(4)
    total = store.write("BTCUSDT", "6h", df)
    assert total == 4

    result = store.read("BTCUSDT", "6h")
    assert len(result) == 4
    assert result["open_time"][0] == datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)


def test_write_merge_dedup(store: ParquetStore):
    df1 = _make_bars(3)
    store.write("BTCUSDT", "6h", df1)

    # Overlapping + new data
    df2 = _make_bars(4)
    total = store.write("BTCUSDT", "6h", df2)
    assert total == 4  # 3 overlap + 1 new


def test_read_empty(store: ParquetStore):
    df = store.read("NONEXISTENT", "6h")
    assert df.is_empty()


def test_has_data(store: ParquetStore):
    assert not store.has_data("BTCUSDT", "6h")
    store.write("BTCUSDT", "6h", _make_bars(2))
    assert store.has_data("BTCUSDT", "6h")


def test_latest_timestamp(store: ParquetStore):
    assert store.latest_timestamp("BTCUSDT", "6h") is None

    df = _make_bars(3)
    store.write("BTCUSDT", "6h", df)
    latest = store.latest_timestamp("BTCUSDT", "6h")
    assert latest == datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)


def test_symbols(store: ParquetStore):
    assert store.symbols() == []
    store.write("BTCUSDT", "6h", _make_bars(1))
    store.write("ETHUSDT", "6h", _make_bars(1))
    assert store.symbols() == ["BTCUSDT", "ETHUSDT"]


def test_coverage(store: ParquetStore):
    cov = store.coverage("BTCUSDT", "6h")
    assert cov["bars"] == 0

    store.write("BTCUSDT", "6h", _make_bars(3))
    cov = store.coverage("BTCUSDT", "6h")
    assert cov["bars"] == 3
    assert cov["symbol"] == "BTCUSDT"


def test_read_with_time_filter(store: ParquetStore):
    df = _make_bars(4)
    store.write("BTCUSDT", "6h", df)

    result = store.read(
        "BTCUSDT",
        "6h",
        start=datetime(2024, 1, 1, 6, 0, 0, tzinfo=UTC),
        end=datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
    )
    assert len(result) == 2
