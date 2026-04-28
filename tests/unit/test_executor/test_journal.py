"""Tests for TradeJournal load functions and live report conversion."""

from __future__ import annotations

import tempfile

from executor.journal import TradeJournal
from executor.routes import _journal_to_report_format


class TestJournalLoad:
    def test_load_empty(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            j = TradeJournal(d)
            assert j.load_equity() == []
            assert j.load_trades() == []
            assert j.load_signals() == []
            assert j.load_targets() == []

    def test_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            j = TradeJournal(d)
            j.record_tick(1, 10000.0, 10000.0, {"BTC": {"side": "long"}})
            j.record_tick(2, 10500.0, 10500.0, {})
            j.record_trade(1, "open_long", "BTC", 0.1, 50000, "signal", side="long")
            j.record_signals(1, {"BTC": 0.5, "ETH": -0.2})
            j.record_target(
                bar=1,
                profile="test-profile",
                target_ts="2024-01-01T00:00:00+00:00",
                staleness_seconds=12.3456,
                target_gross=0.3,
                normalized_gross=0.2,
                weights={"BTC": 0.123456789},
                orders=[{"symbol": "BTC", "side": "buy", "qty": 0.1}],
                ignored_weights={"XRPUSDT": 0.050000004},
            )

            assert len(j.load_equity()) == 2
            assert len(j.load_trades()) == 1
            assert len(j.load_signals()) == 1
            targets = j.load_targets()
            assert len(targets) == 1
            assert targets[0]["profile"] == "test-profile"
            assert targets[0]["weights"] == {"BTC": 0.12345679}
            assert targets[0]["ignored_weights"] == {"XRPUSDT": 0.05}
            assert targets[0]["ignored_gross"] == 0.05
            assert targets[0]["ignored_gross_ratio"] == 0.166667
            assert targets[0]["execution_coverage"] == 0.666667


class TestJournalToReportFormat:
    def test_empty_journal(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            j = TradeJournal(d)
            result = _journal_to_report_format(j)
            assert result["status"] == "error"

    def test_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            j = TradeJournal(d)
            j.record_tick(1, 10000.0, 10000.0, {"BTC": {"side": "long"}})
            j.record_tick(2, 10200.0, 10200.0, {"BTC": {"side": "long"}})
            j.record_trade(1, "open_long", "BTC", 0.1, 50000, "signal", side="long")
            j.record_trade(
                2,
                "close",
                "BTC",
                0.1,
                52000,
                "tp",
                side="long",
                entry_price=50000,
                pnl=200,
                pnl_pct=4.0,
            )

            result = _journal_to_report_format(j)
            assert len(result["equity_curve"]) == 2
            assert len(result["trades"]) == 1  # only closed trade has pnl
            assert result["bars"] == 2
            assert result["positions"] == {"BTC": {"side": "long"}}
