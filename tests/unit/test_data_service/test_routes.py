"""Tests for data-service REST API routes."""

from __future__ import annotations

import pytest
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
