"""Hyperliquid account-state accounting tests."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from exchange.hyperliquid import HyperliquidAdapter


class FakeInfo:
    def __init__(
        self,
        *,
        account_mode: str,
        user_state: dict[str, Any],
        spot_state: dict[str, Any],
    ) -> None:
        self._account_mode = account_mode
        self._user_state = user_state
        self._spot_state = spot_state

    def user_state(self, _address: str) -> dict[str, Any]:
        return self._user_state

    def spot_user_state(self, _address: str) -> dict[str, Any]:
        return self._spot_state

    def query_user_abstraction_state(self, _address: str) -> str:
        return self._account_mode


def make_adapter(info: FakeInfo) -> HyperliquidAdapter:
    adapter = HyperliquidAdapter.__new__(HyperliquidAdapter)
    adapter._info = info
    adapter._address = "0xabc"
    adapter._sz_decimals = {}

    async def fake_run_sync(func: Any, *args: Any, timeout: float = 15.0) -> Any:
        return func(*args)

    adapter._run_sync = fake_run_sync  # type: ignore[method-assign]
    return adapter


def test_format_size_rounds_down_to_asset_precision() -> None:
    adapter = HyperliquidAdapter.__new__(HyperliquidAdapter)
    adapter._sz_decimals = {"NEAR": 1, "PURR": 0}

    assert adapter._format_size(Decimal("13.85980505146829"), "NEAR") == 13.8
    assert adapter._format_size(Decimal("2.9"), "PURR") == 2.0
    assert adapter._format_size(Decimal("0.009"), "BTC") == 0.0


async def test_unified_account_uses_spot_balance_and_unrealized_pnl() -> None:
    adapter = make_adapter(
        FakeInfo(
            account_mode="unifiedAccount",
            user_state={
                "marginSummary": {
                    # These per-dex values are not the account-level web balance
                    # in unified mode and must not be summed with spot available.
                    "accountValue": "28.06196",
                    "totalRawUsd": "-126.453402",
                    "totalMarginUsed": "30.903072",
                },
                "assetPositions": [
                    {
                        "position": {
                            "coin": "BTC",
                            "szi": "0.1",
                            "entryPx": "100",
                            "unrealizedPnl": "1.5",
                            "leverage": {"value": 5},
                        }
                    },
                    {
                        "position": {
                            "coin": "SOL",
                            "szi": "2",
                            "entryPx": "10",
                            "unrealizedPnl": "-3.0",
                            "leverage": {"value": 5},
                        }
                    },
                ],
            },
            spot_state={
                "balances": [
                    {"coin": "USDC", "total": "100.180492", "hold": "30.903072"}
                ]
            },
        )
    )

    state = await adapter.get_account_state()

    assert state.available_balance == Decimal("69.277420")
    assert state.unrealized_pnl == Decimal("-1.5")
    assert state.equity == Decimal("100.180492")
    assert state.total_margin_used == Decimal("30.903072")
    assert [p.symbol for p in state.positions] == ["BTC", "SOL"]


async def test_non_unified_account_uses_perp_margin_summary() -> None:
    adapter = make_adapter(
        FakeInfo(
            account_mode="normal",
            user_state={
                "marginSummary": {
                    "accountValue": "2000.5",
                    "totalRawUsd": "123.4",
                    "totalMarginUsed": "10",
                },
                "assetPositions": [],
            },
            spot_state={"balances": [{"coin": "USDC", "total": "100", "hold": "1"}]},
        )
    )

    state = await adapter.get_account_state()

    assert state.equity == Decimal("2000.5")
    assert state.available_balance == Decimal("123.4")
    assert state.unrealized_pnl == Decimal("0")


async def test_fetch_spot_balance_retries_on_transient_failure() -> None:
    """_fetch_spot_balance with @RETRY_TRANSIENT retries 3 times then succeeds."""
    call_count = 0

    class FlakySpotInfo(FakeInfo):
        def spot_user_state(self, _address: str) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("transient HL spot timeout")
            return self._spot_state

    adapter = make_adapter(
        FlakySpotInfo(
            account_mode="unifiedAccount",
            user_state={
                "marginSummary": {
                    "accountValue": "10",
                    "totalRawUsd": "0",
                    "totalMarginUsed": "0",
                },
                "assetPositions": [],
            },
            spot_state={"balances": [{"coin": "USDC", "total": "200", "hold": "5"}]},
        )
    )

    state = await adapter.get_account_state()

    assert call_count == 3
    assert state.equity == Decimal("200")
