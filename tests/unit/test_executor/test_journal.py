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

    def test_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            j = TradeJournal(d)
            j.record_tick(1, 10000.0, 10000.0, {"BTC": {"side": "long"}})
            j.record_tick(2, 10500.0, 10500.0, {})
            j.record_trade(
                1, "open_long", "BTC", 0.1, 50000, "signal", side="long"
            )
            j.record_signals(1, {"BTC": 0.5, "ETH": -0.2})

            assert len(j.load_equity()) == 2
            assert len(j.load_trades()) == 1
            assert len(j.load_signals()) == 1


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
            j.record_trade(
                1, "open_long", "BTC", 0.1, 50000, "signal", side="long"
            )
            j.record_trade(
                2, "close", "BTC", 0.1, 52000, "tp",
                side="long", entry_price=50000, pnl=200, pnl_pct=4.0,
            )

            result = _journal_to_report_format(j)
            assert len(result["equity_curve"]) == 2
            assert len(result["trades"]) == 1  # only closed trade has pnl
            assert result["bars"] == 2
            assert result["positions"] == {"BTC": {"side": "long"}}
