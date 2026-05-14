"""Tests for TradeJournal load functions and live report conversion."""

from __future__ import annotations

import tempfile

import pytest
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

            trades = j.load_trades()
            assert len(j.load_equity()) == 2
            assert len(trades) == 1
            assert trades[0]["price"] == 50000.0
            assert len(j.load_signals()) == 1
            targets = j.load_targets()
            assert len(targets) == 1
            assert targets[0]["profile"] == "test-profile"
            assert targets[0]["weights"] == {"BTC": 0.123456789}
            assert targets[0]["ignored_weights"] == {"XRPUSDT": 0.050000004}
            assert targets[0]["ignored_gross"] == 0.050000004
            assert targets[0]["ignored_gross_ratio"] == pytest.approx(0.16666668)
            assert targets[0]["execution_coverage"] == pytest.approx(0.6666666666666667)

    def test_identity_fields_are_additive_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            j = TradeJournal(
                d,
                live_instance_id="cta-forge-mainnet-pilot-01",
                run_id="20260514T221212Z-deadbeef",
                public_instance_slug="mainnet-pilot",
            )
            j.record_tick(1, 101.123456789, 102.0, {})
            j.record_trade(
                1,
                "open_long",
                "SEI",
                259.123456789,
                0.068897,
                "target",
                side="long",
            )
            j.record_signals(1, {"SEI": 0.123456789})
            j.record_target(
                bar=1,
                profile="test-profile",
                target_ts="2026-05-14T22:00:00+00:00",
                staleness_seconds=1.23456789,
                target_gross=0.123456789,
                normalized_gross=0.123456789,
                weights={"SEI": 0.123456789},
                orders=[{"symbol": "SEI", "qty": 259.123456789}],
            )

            for record in (
                j.load_equity()[0],
                j.load_trades()[0],
                j.load_signals()[0],
                j.load_targets()[0],
            ):
                assert record["live_instance_id"] == "cta-forge-mainnet-pilot-01"
                assert record["run_id"] == "20260514T221212Z-deadbeef"
                assert record["public_instance_slug"] == "mainnet-pilot"

            assert j.load_trades()[0]["qty"] == 259.123456789
            assert j.load_targets()[0]["weights"] == {"SEI": 0.123456789}

    def test_identity_fields_are_absent_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            j = TradeJournal(d)
            j.record_tick(1, 100.0, 100.0, {})

            record = j.load_equity()[0]
            assert "live_instance_id" not in record
            assert "run_id" not in record
            assert "public_instance_slug" not in record

    def test_record_trade_preserves_price_precision(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            j = TradeJournal(d)
            j.record_trade(
                1,
                "close",
                "SEI",
                259.123456789,
                0.068897,
                "target",
                side="long",
                entry_price=0.059547,
                pnl=1.234567,
                pnl_pct=2.345678,
            )

            trade = j.load_trades()[0]
            assert trade["qty"] == 259.123456789
            assert trade["price"] == 0.068897
            assert trade["entry_price"] == 0.059547
            assert trade["pnl"] == 1.234567
            assert trade["pnl_pct"] == 2.345678

    def test_record_tick_clamps_stale_peak_to_current_equity(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            j = TradeJournal(d)
            j.record_tick(1, 101.0, 100.0, {})

            record = j.load_equity()[0]
            assert record["peak"] == 101.0
            assert record["dd_pct"] == 0.0

    def test_load_skips_corrupt_jsonl_lines(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            j = TradeJournal(d)
            j.record_tick(1, 100.0, 100.0, {})
            (j._equity_file).write_text(
                j._equity_file.read_text()
                + "{not valid json}\n"
                + '{"bar": 2, "equity": 101.0}\n'
            )

            records = j.load_equity()

            assert [record["bar"] for record in records] == [1, 2]


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
            assert len(result["equity_records"]) == 2
            assert len(result["equity_curve"]) == 2
            assert len(result["trades"]) == 1  # only closed trade has pnl
            assert result["bars"] == 2
            assert result["positions"] == {"BTC": {"side": "long"}}
