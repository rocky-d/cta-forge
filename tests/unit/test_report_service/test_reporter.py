"""Tests for reporter metrics and routes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
from fastapi.testclient import TestClient
from report_service.app import app
from report_service.metrics import calculate_metrics


def _make_curve(n: int = 100, trend: float = 0.001) -> list[tuple[datetime, float]]:
    np.random.seed(42)
    curve = []
    equity = 10000.0
    for i in range(n):
        equity = equity * (1 + trend + np.random.normal(0, 0.01))
        curve.append(
            (datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i * 6), equity)
        )
    return curve


class TestMetrics:
    def test_basic_metrics(self):
        curve = _make_curve(200, trend=0.002)
        trades = [{"pnl": 100}, {"pnl": -50}, {"pnl": 75}]
        metrics = calculate_metrics(curve, trades)

        assert metrics.total_return > 0
        assert metrics.sharpe_ratio > 0
        assert metrics.max_drawdown >= 0
        assert metrics.num_trades == 3
        assert metrics.win_rate > 0.5

    def test_empty_curve(self):
        metrics = calculate_metrics([], [])
        assert metrics.total_return == 0.0

    def test_no_trades(self):
        curve = _make_curve(50)
        metrics = calculate_metrics(curve, [])
        assert metrics.num_trades == 0
        assert metrics.win_rate == 0.0

    def test_profit_factor(self):
        trades = [{"pnl": 100}, {"pnl": 100}, {"pnl": -50}]
        metrics = calculate_metrics(_make_curve(50), trades)
        assert metrics.profit_factor == 4.0  # 200 / 50


class TestRoutes:
    def setup_method(self):
        self.client = TestClient(app)

    def test_health(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200

    def test_report_endpoint(self):
        curve = [(str(t), e) for t, e in _make_curve(50)]
        resp = self.client.post(
            "/report",
            json={
                "equity_curve": curve,
                "trades": [{"pnl": 100}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "metrics" in data
        assert "raw" in data

    def test_plot_endpoint(self):
        curve = [(str(t), e) for t, e in _make_curve(50)]
        resp = self.client.post(
            "/plot",
            json={
                "equity_curve": curve,
                "chart_type": "equity",
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_plot_base64_endpoint(self):
        curve = [(str(t), e) for t, e in _make_curve(50)]
        resp = self.client.post(
            "/plot/base64",
            json={
                "equity_curve": curve,
                "chart_type": "drawdown",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "image" in data
        assert data["media_type"] == "image/png"
