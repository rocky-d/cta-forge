"""Tests for report-service metrics and routes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient
from report_service.app import app
from report_service.metrics import calculate_live_metrics, calculate_metrics
from report_service.plot import plot_live_journal


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

    def test_live_metrics_use_elapsed_time_and_suppress_short_samples(self):
        curve = [
            (datetime(2026, 5, 5, 0, tzinfo=UTC), 100.0),
            (datetime(2026, 5, 5, 6, tzinfo=UTC), 105.0),
        ]
        live_metrics = calculate_live_metrics(curve, [])

        assert live_metrics.annualized_status == "unstable"
        assert live_metrics.metrics.total_return == pytest.approx(0.05)
        assert live_metrics.metrics.annualized_return == 0.0
        assert live_metrics.annualized_return_raw is not None
        assert live_metrics.cadence_median_hours == 6.0

    def test_live_metrics_suppresses_short_sample_after_one_week(self):
        curve = [
            (datetime(2026, 5, 1, tzinfo=UTC), 100.0),
            (datetime(2026, 5, 11, tzinfo=UTC), 101.0),
        ]
        live_metrics = calculate_live_metrics(curve, [])

        assert live_metrics.annualized_status == "short_sample"
        assert live_metrics.metrics.annualized_return == 0.0
        assert live_metrics.annualized_return_raw is not None
        assert live_metrics.annualized_return_raw > 0
        assert live_metrics.elapsed_days == 10.0


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

    def test_live_journal_plot_ignores_stale_recorded_drawdown(self):
        rows = [
            {
                "ts": "2026-05-05T14:00:00+00:00",
                "bar": 1,
                "equity": 100.0,
                "peak": 100.0,
                "dd_pct": 0.0,
            },
            {
                "ts": "2026-05-05T15:00:00+00:00",
                "bar": 2,
                "equity": 101.0,
                "peak": 100.0,
                "dd_pct": -1.0,
            },
        ]
        img_bytes = plot_live_journal(rows, [{"kind": "target_buy"}])
        assert img_bytes.startswith(b"\x89PNG")

    def test_live_journal_plot_endpoint(self):
        rows = [
            {
                "ts": "2026-05-05T14:00:00+00:00",
                "bar": 1,
                "equity": 100.0,
                "peak": 100.0,
                "dd_pct": 0.0,
            },
            {
                "ts": "2026-05-05T15:00:00+00:00",
                "bar": 2,
                "equity": 99.0,
                "peak": 100.0,
                "dd_pct": 1.0,
            },
        ]
        resp = self.client.post(
            "/plot/live-journal",
            json={"equity_records": rows, "trades": []},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
