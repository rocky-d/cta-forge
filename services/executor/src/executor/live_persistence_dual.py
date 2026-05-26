"""Dual-write helpers for live persistence shadow mode.

This module is intentionally not wired into the live runtime yet. It provides
small, testable wrappers for the future phase where file-backed persistence
remains the source of truth while PostgreSQL receives shadow writes.
"""

from __future__ import annotations

import logging
from typing import Literal, cast

from .journal import LiveJournalStore
from .live import LiveState
from .state import LiveStateStore

logger = logging.getLogger(__name__)

PersistenceBackend = Literal["file", "dual", "postgres"]
ShadowWriteFailurePolicy = Literal["warn", "raise"]


def parse_persistence_backend(value: str | None) -> PersistenceBackend:
    """Parse the runtime persistence backend flag.

    The safe default is the current file-backed behavior. Invalid values fail
    closed so a typo cannot silently enable a different persistence path.
    """

    if value is None or not value.strip():
        return "file"
    normalized = value.strip().lower()
    if normalized in {"file", "dual", "postgres"}:
        return cast(PersistenceBackend, normalized)
    msg = (
        f"invalid PERSISTENCE_BACKEND {value!r}; expected one of: file, dual, postgres"
    )
    raise ValueError(msg)


def _validate_shadow_failure_policy(policy: str) -> ShadowWriteFailurePolicy:
    if policy in {"warn", "raise"}:
        return cast(ShadowWriteFailurePolicy, policy)
    msg = f"invalid shadow write failure policy {policy!r}; expected warn or raise"
    raise ValueError(msg)


class DualLiveJournalStore:
    """File-first dual journal store.

    The primary store remains the read/source-of-truth path. The shadow store is
    written only after the primary write succeeds.
    """

    def __init__(
        self,
        primary: LiveJournalStore,
        shadow: LiveJournalStore,
        *,
        shadow_failure_policy: ShadowWriteFailurePolicy = "warn",
    ) -> None:
        self._primary = primary
        self._shadow = shadow
        self._shadow_failure_policy = _validate_shadow_failure_policy(
            shadow_failure_policy
        )

    def record_tick(
        self,
        bar: int,
        equity: float,
        peak_equity: float,
        positions: dict[str, dict],
    ) -> None:
        self._primary.record_tick(bar, equity, peak_equity, positions)
        self._write_shadow("record_tick", bar, equity, peak_equity, positions)

    def record_trade(
        self,
        bar: int,
        kind: str,
        symbol: str,
        qty: float,
        price: float,
        reason: str,
        *,
        side: str = "",
        entry_price: float = 0.0,
        pnl: float = 0.0,
        pnl_pct: float = 0.0,
        held_bars: int = 0,
        exchange_order_id: str | None = None,
    ) -> None:
        self._primary.record_trade(
            bar,
            kind,
            symbol,
            qty,
            price,
            reason,
            side=side,
            entry_price=entry_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            held_bars=held_bars,
            exchange_order_id=exchange_order_id,
        )
        self._write_shadow(
            "record_trade",
            bar,
            kind,
            symbol,
            qty,
            price,
            reason,
            side=side,
            entry_price=entry_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            held_bars=held_bars,
            exchange_order_id=exchange_order_id,
        )

    def record_signals(self, bar: int, signals: dict[str, float]) -> None:
        self._primary.record_signals(bar, signals)
        self._write_shadow("record_signals", bar, signals)

    def record_target(
        self,
        *,
        bar: int,
        profile: str,
        target_ts: str,
        staleness_seconds: float,
        target_gross: float,
        normalized_gross: float,
        weights: dict[str, float],
        orders: list[dict],
        ignored_weights: dict[str, float] | None = None,
        submitted_orders: list[dict] | None = None,
        filled_trades: list[dict] | None = None,
        failed_orders: list[dict] | None = None,
    ) -> None:
        self._primary.record_target(
            bar=bar,
            profile=profile,
            target_ts=target_ts,
            staleness_seconds=staleness_seconds,
            target_gross=target_gross,
            normalized_gross=normalized_gross,
            weights=weights,
            orders=orders,
            ignored_weights=ignored_weights,
            submitted_orders=submitted_orders,
            filled_trades=filled_trades,
            failed_orders=failed_orders,
        )
        self._write_shadow(
            "record_target",
            bar=bar,
            profile=profile,
            target_ts=target_ts,
            staleness_seconds=staleness_seconds,
            target_gross=target_gross,
            normalized_gross=normalized_gross,
            weights=weights,
            orders=orders,
            ignored_weights=ignored_weights,
            submitted_orders=submitted_orders,
            filled_trades=filled_trades,
            failed_orders=failed_orders,
        )

    def load_equity(self) -> list[dict]:
        """Load from the primary store only."""

        return self._primary.load_equity()

    def load_trades(self) -> list[dict]:
        """Load from the primary store only."""

        return self._primary.load_trades()

    def load_signals(self) -> list[dict]:
        """Load from the primary store only."""

        return self._primary.load_signals()

    def load_targets(self) -> list[dict]:
        """Load from the primary store only."""

        return self._primary.load_targets()

    def _write_shadow(self, operation: str, *args: object, **kwargs: object) -> None:
        try:
            getattr(self._shadow, operation)(*args, **kwargs)
        except Exception:
            logger.exception("shadow live journal write failed: %s", operation)
            if self._shadow_failure_policy == "raise":
                raise


class DualLiveStateStore:
    """File-first dual checkpoint store.

    Reads always use the primary store. Saves write primary first and shadow
    second, matching the intended shadow-mode source-of-truth boundary.
    """

    def __init__(
        self,
        primary: LiveStateStore,
        shadow: LiveStateStore,
        *,
        shadow_failure_policy: ShadowWriteFailurePolicy = "warn",
    ) -> None:
        self._primary = primary
        self._shadow = shadow
        self._shadow_failure_policy = _validate_shadow_failure_policy(
            shadow_failure_policy
        )

    def load(self) -> LiveState | None:
        """Load from the primary store only."""

        return self._primary.load()

    def save(self, state: LiveState) -> None:
        """Save primary first, then shadow."""

        self._primary.save(state)
        try:
            self._shadow.save(state)
        except Exception:
            logger.exception("shadow live checkpoint write failed: save")
            if self._shadow_failure_policy == "raise":
                raise
