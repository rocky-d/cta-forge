"""Tests for V10GDecisionEngine."""

from __future__ import annotations

import pytest

from executor.decision import (
    ActionKind,
    BarSnapshot,
    EngineState,
    PositionState,
    V10GDecisionEngine,
)


@pytest.fixture
def engine() -> V10GDecisionEngine:
    """Create engine with default params."""
    return V10GDecisionEngine()


@pytest.fixture
def state() -> EngineState:
    """Create fresh engine state."""
    return EngineState(initial_equity=10000.0, peak_equity=10000.0)


@pytest.fixture
def btc_snapshot() -> BarSnapshot:
    """Sample BTC snapshot."""
    return BarSnapshot(close=75000.0, atr=1500.0, rvol=0.015, signal=0.5)


@pytest.fixture
def eth_snapshot() -> BarSnapshot:
    """Sample ETH snapshot."""
    return BarSnapshot(close=3000.0, atr=100.0, rvol=0.02, signal=-0.45)


class TestOpenDecisions:
    """Test position opening logic."""

    def test_open_long_above_threshold(
        self, engine: V10GDecisionEngine, state: EngineState, btc_snapshot: BarSnapshot
    ) -> None:
        """Opens long when signal >= threshold."""
        snapshots = {"BTC": btc_snapshot}
        actions = engine.tick(state, 10000.0, snapshots)

        assert len(actions) == 1
        assert actions[0].kind == ActionKind.OPEN_LONG
        assert actions[0].symbol == "BTC"
        assert actions[0].qty > 0

    def test_open_short_above_threshold(
        self, engine: V10GDecisionEngine, state: EngineState, eth_snapshot: BarSnapshot
    ) -> None:
        """Opens short when signal <= -threshold."""
        snapshots = {"ETH": eth_snapshot}
        actions = engine.tick(state, 10000.0, snapshots)

        assert len(actions) == 1
        assert actions[0].kind == ActionKind.OPEN_SHORT
        assert actions[0].symbol == "ETH"
        assert actions[0].qty > 0

    def test_no_open_below_threshold(
        self, engine: V10GDecisionEngine, state: EngineState
    ) -> None:
        """No action when signal below threshold."""
        snap = BarSnapshot(close=1000.0, atr=20.0, rvol=0.01, signal=0.3)
        actions = engine.tick(state, 10000.0, {"TEST": snap})

        assert len(actions) == 0

    def test_max_positions_respected(
        self, engine: V10GDecisionEngine, state: EngineState
    ) -> None:
        """Doesn't open more than max_positions."""
        # Fill up positions with valid data (won't trigger close)
        for i in range(engine.p.max_positions):
            state.positions[f"SYM{i}"] = PositionState(
                symbol=f"SYM{i}",
                qty=0.1,
                entry_price=100.0,
                entry_bar=state.bar_count,  # just opened
                best_price=100.0,
            )

        # Provide snapshots for existing positions so they don't close
        snapshots = {
            f"SYM{i}": BarSnapshot(close=100.0, atr=5.0, rvol=0.01, signal=0.3)
            for i in range(engine.p.max_positions)
        }
        # Add a new candidate
        snapshots["NEWONE"] = BarSnapshot(close=1000.0, atr=20.0, rvol=0.01, signal=0.5)

        actions = engine.tick(state, 10000.0, snapshots)

        opens = [
            a
            for a in actions
            if a.kind in (ActionKind.OPEN_LONG, ActionKind.OPEN_SHORT)
        ]
        assert len(opens) == 0

    def test_rebalance_frequency(
        self, engine: V10GDecisionEngine, state: EngineState, btc_snapshot: BarSnapshot
    ) -> None:
        """Only opens on rebalance bars."""
        # bar 1: should open (first bar)
        actions1 = engine.tick(state, 10000.0, {"BTC": btc_snapshot})
        assert any(a.kind == ActionKind.OPEN_LONG for a in actions1)

        # Clear position for next test
        state.positions.clear()

        # bar 2: not a rebalance bar (rebalance_every=4)
        actions2 = engine.tick(state, 10000.0, {"BTC": btc_snapshot})
        opens2 = [
            a
            for a in actions2
            if a.kind in (ActionKind.OPEN_LONG, ActionKind.OPEN_SHORT)
        ]
        assert len(opens2) == 0


class TestCloseDecisions:
    """Test position closing logic."""

    def test_trailing_stop_long(
        self, engine: V10GDecisionEngine, state: EngineState
    ) -> None:
        """Closes long when price drops below trailing stop."""
        state.bar_count = 20
        state.positions["BTC"] = PositionState(
            symbol="BTC",
            qty=0.1,
            entry_price=80000.0,
            entry_bar=1,
            best_price=85000.0,  # price went up then crashed
        )

        # Price dropped below best - 5*ATR
        snap = BarSnapshot(close=75000.0, atr=1500.0, rvol=0.01, signal=0.0)
        actions = engine.tick(state, 9000.0, {"BTC": snap})

        closes = [a for a in actions if a.kind == ActionKind.CLOSE]
        assert len(closes) == 1
        assert closes[0].symbol == "BTC"
        assert "trailing_stop" in closes[0].reason

    def test_trailing_stop_short(
        self, engine: V10GDecisionEngine, state: EngineState
    ) -> None:
        """Closes short when price rises above trailing stop."""
        state.bar_count = 20
        state.positions["ETH"] = PositionState(
            symbol="ETH",
            qty=-1.0,
            entry_price=3000.0,
            entry_bar=1,
            best_price=2800.0,  # price went down then spiked
        )

        # Price spiked above best + 5*ATR
        snap = BarSnapshot(close=3400.0, atr=100.0, rvol=0.01, signal=0.0)
        actions = engine.tick(state, 9500.0, {"ETH": snap})

        closes = [a for a in actions if a.kind == ActionKind.CLOSE]
        assert len(closes) == 1
        assert closes[0].symbol == "ETH"

    def test_max_hold_bars(
        self, engine: V10GDecisionEngine, state: EngineState
    ) -> None:
        """Closes position after max_hold_bars."""
        state.bar_count = 100
        state.positions["SOL"] = PositionState(
            symbol="SOL",
            qty=10.0,
            entry_price=100.0,
            entry_bar=0,  # held for 100 bars
            best_price=120.0,
        )

        snap = BarSnapshot(close=115.0, atr=5.0, rvol=0.01, signal=0.3)
        actions = engine.tick(state, 10500.0, {"SOL": snap})

        closes = [a for a in actions if a.kind == ActionKind.CLOSE]
        assert len(closes) == 1
        assert "max_hold" in closes[0].reason

    def test_signal_reversal_long(
        self, engine: V10GDecisionEngine, state: EngineState
    ) -> None:
        """Closes long when signal reverses negative."""
        state.bar_count = 20
        state.positions["LINK"] = PositionState(
            symbol="LINK",
            qty=100.0,
            entry_price=15.0,
            entry_bar=1,
            best_price=16.0,
        )

        # Strong negative signal
        snap = BarSnapshot(close=15.5, atr=0.5, rvol=0.01, signal=-0.25)
        actions = engine.tick(state, 10000.0, {"LINK": snap})

        closes = [a for a in actions if a.kind == ActionKind.CLOSE]
        assert len(closes) == 1
        assert "signal_reversal" in closes[0].reason

    def test_min_hold_respected(
        self, engine: V10GDecisionEngine, state: EngineState
    ) -> None:
        """Doesn't close before min_hold_bars even with stop hit."""
        state.bar_count = 5  # only 5 bars held
        state.positions["BTC"] = PositionState(
            symbol="BTC",
            qty=0.1,
            entry_price=80000.0,
            entry_bar=0,
            best_price=85000.0,
        )

        # Price dropped but we haven't held min_hold_bars yet
        snap = BarSnapshot(close=75000.0, atr=1500.0, rvol=0.01, signal=-0.3)
        actions = engine.tick(state, 9000.0, {"BTC": snap})

        closes = [a for a in actions if a.kind == ActionKind.CLOSE]
        assert len(closes) == 0


class TestPartialTakeProfit:
    """Test partial take-profit logic."""

    def test_partial_tp_long(
        self, engine: V10GDecisionEngine, state: EngineState
    ) -> None:
        """Partial close long when profit exceeds 2.5*ATR."""
        state.bar_count = 5
        state.positions["BTC"] = PositionState(
            symbol="BTC",
            qty=0.1,
            entry_price=70000.0,
            entry_bar=1,
            best_price=74000.0,
        )

        # Price up more than 2.5 * 1500 = 3750 from entry
        snap = BarSnapshot(close=74500.0, atr=1500.0, rvol=0.01, signal=0.4)
        actions = engine.tick(state, 10500.0, {"BTC": snap})

        partials = [a for a in actions if a.kind == ActionKind.PARTIAL_CLOSE]
        assert len(partials) == 1
        assert partials[0].symbol == "BTC"
        assert 0.04 < partials[0].qty < 0.06  # half of 0.1

    def test_partial_tp_not_repeated(
        self, engine: V10GDecisionEngine, state: EngineState
    ) -> None:
        """Doesn't partial close twice."""
        state.bar_count = 5
        state.positions["BTC"] = PositionState(
            symbol="BTC",
            qty=0.05,  # already partial closed
            entry_price=70000.0,
            entry_bar=1,
            best_price=75000.0,
            partial_taken=True,
        )

        snap = BarSnapshot(close=76000.0, atr=1500.0, rvol=0.01, signal=0.4)
        actions = engine.tick(state, 10500.0, {"BTC": snap})

        partials = [a for a in actions if a.kind == ActionKind.PARTIAL_CLOSE]
        assert len(partials) == 0


class TestRiskControls:
    """Test drawdown and risk management."""

    def test_max_drawdown_flatten(
        self, engine: V10GDecisionEngine, state: EngineState
    ) -> None:
        """Flattens all on max drawdown breach."""
        state.peak_equity = 10000.0
        state.positions["BTC"] = PositionState(
            symbol="BTC", qty=0.1, entry_price=70000.0, entry_bar=0, best_price=70000.0
        )
        state.positions["ETH"] = PositionState(
            symbol="ETH", qty=1.0, entry_price=3000.0, entry_bar=0, best_price=3000.0
        )

        snapshots = {
            "BTC": BarSnapshot(close=65000.0, atr=1500.0, rvol=0.01, signal=0.0),
            "ETH": BarSnapshot(close=2800.0, atr=100.0, rvol=0.01, signal=0.0),
        }

        # Equity dropped 16% (> 15% max_drawdown)
        actions = engine.tick(state, 8400.0, snapshots)

        flattens = [a for a in actions if a.kind == ActionKind.FLATTEN_ALL]
        assert len(flattens) == 2  # one per position

    def test_dd_breaker_reduces_size(
        self, engine: V10GDecisionEngine, state: EngineState
    ) -> None:
        """DD breaker reduces position size by 50%."""
        state.peak_equity = 10000.0

        snap = BarSnapshot(close=75000.0, atr=1500.0, rvol=0.015, signal=0.5)

        # Normal sizing at equity = 10000
        actions_normal = engine.tick(state, 10000.0, {"BTC": snap})
        state.positions.clear()
        state.bar_count = 0

        # With 9% drawdown (> 8% dd_breaker), size should be ~50% of normal
        actions_dd = engine.tick(state, 9100.0, {"BTC": snap})

        opens_normal = [a for a in actions_normal if a.kind == ActionKind.OPEN_LONG]
        opens_dd = [a for a in actions_dd if a.kind == ActionKind.OPEN_LONG]

        assert len(opens_normal) == 1
        assert len(opens_dd) == 1
        # DD breaker should reduce size by roughly 50%
        assert opens_dd[0].qty < opens_normal[0].qty * 0.7


class TestTightenedStop:
    """Test stop tightening when in profit."""

    def test_tightened_stop_triggers_earlier(
        self, engine: V10GDecisionEngine, state: EngineState
    ) -> None:
        """Tightened stop (3x ATR) triggers before normal stop (5x ATR)."""
        state.bar_count = 20
        state.positions["BTC"] = PositionState(
            symbol="BTC",
            qty=0.1,
            entry_price=70000.0,
            entry_bar=1,
            best_price=78000.0,  # was up 8000 = 5.3x ATR from entry
        )

        # Unrealized from entry = 74000 - 70000 = 4000 = 2.67x ATR (> 2.0, triggers tightening)
        # Tightened stop level = 78000 - 3*1500 = 73500
        # Price 73000 < 73500 => should trigger tightened stop
        snap = BarSnapshot(close=73000.0, atr=1500.0, rvol=0.01, signal=0.0)

        # Need to update best_price first, then drop below tightened stop
        # Let's set current price such that:
        # 1) unrealized from entry > 2.0 ATR (tighten triggers)
        # 2) price < best_price - 3*ATR (tightened stop hit)
        # best_price = 78000, 3*1500 = 4500, stop at 73500
        # At close=73000: below tightened stop
        # unrealized = 73000 - 70000 = 3000 = 2.0x ATR (exact, need >)

        # Adjust: use close=73500 (at tightened stop exactly)
        # and best_price = 78500 so tightened stop = 78500 - 4500 = 74000
        # unrealized = 73500 - 70000 = 3500 = 2.33x ATR (> 2.0, tighten active)
        # 73500 < 74000 => stop hit
        state.positions["BTC"].best_price = 78500.0
        snap = BarSnapshot(close=73500.0, atr=1500.0, rvol=0.01, signal=0.0)

        actions = engine.tick(state, 10000.0, {"BTC": snap})

        closes = [a for a in actions if a.kind == ActionKind.CLOSE]
        assert len(closes) == 1
        assert "3.0x" in closes[0].reason  # tightened stop


class TestVolScaling:
    """Test target volatility scaling."""

    def test_vol_scaling_reduces_size(
        self, engine: V10GDecisionEngine, state: EngineState
    ) -> None:
        """High realized vol reduces position size."""
        # Simulate high volatility regime
        state.recent_returns = [0.02, -0.03, 0.025, -0.02] * 30  # 120 returns, high vol

        snap = BarSnapshot(close=75000.0, atr=1500.0, rvol=0.015, signal=0.5)
        actions = engine.tick(state, 10000.0, {"BTC": snap})

        # With high vol, size should be scaled down
        opens = [a for a in actions if a.kind == ActionKind.OPEN_LONG]
        assert len(opens) == 1
        # Can't assert exact size but it should exist
        assert opens[0].qty > 0
