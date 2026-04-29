"""Tests for data-service REST API routes."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest
from data_service import routes
from data_service.app import app
from data_service.store import ParquetStore
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path):
    app.state.store = ParquetStore(tmp_path / "data")
    return TestClient(app)


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_list_symbols_empty(client: TestClient):
    resp = client.get("/symbols")
    assert resp.status_code == 200
    assert resp.json()["local"] == []


def test_get_bars_empty(client: TestClient):
    resp = client.get("/bars/BTCUSDT?tf=6h")
    assert resp.status_code == 200
    data = resp.json()
    assert data["bars"] == 0
    assert data["data"] == []


def test_status_empty(client: TestClient):
    resp = client.get("/status?tf=6h")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_symbols"] == 0


def test_sync_refetches_latest_open_time(client: TestClient, monkeypatch):
    latest = datetime(2024, 1, 1, 6, tzinfo=UTC)
    app.state.store.write(
        "BTCUSDT",
        "6h",
        pl.DataFrame(
            [
                {
                    "open_time": latest,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1.0,
                    "close_time": datetime(2024, 1, 1, 11, 59, 59, tzinfo=UTC),
                    "quote_volume": 100.0,
                    "trades": 1,
                    "taker_buy_volume": 0.5,
                    "taker_buy_quote_volume": 50.0,
                }
            ]
        ),
    )
    calls = []

    async def fake_fetch_all_klines(
        client, *, symbol, interval, start_ms=None, **kwargs
    ):
        calls.append({"symbol": symbol, "interval": interval, "start_ms": start_ms})
        return pl.DataFrame()

    monkeypatch.setattr(routes.fetcher, "fetch_all_klines", fake_fetch_all_klines)

    resp = client.post("/sync?tf=6h", json=["BTCUSDT"])

    assert resp.status_code == 200
    assert calls == [
        {
            "symbol": "BTCUSDT",
            "interval": "6h",
            "start_ms": int(latest.timestamp() * 1000),
        }
    ]
