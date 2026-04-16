"""Tests for notification backends."""

from __future__ import annotations


import pytest

from executor.notify import (
    LarkNotifier,
    MultiNotifier,
    NullNotifier,
    TelegramNotifier,
)


@pytest.mark.asyncio
async def test_null_notifier_is_silent() -> None:
    """NullNotifier.send() completes without error."""
    n = NullNotifier()
    await n.send("test message")


@pytest.mark.asyncio
async def test_telegram_notifier_handles_failure(monkeypatch) -> None:
    """TelegramNotifier swallows network errors."""
    n = TelegramNotifier("fake-token", "fake-chat")

    async def _mock_post(*args, **kwargs):
        raise ConnectionError("no network")

    # Patch httpx.AsyncClient to fail
    import httpx

    class FailClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def post(self, *args, **kwargs):
            raise ConnectionError("no network")

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FailClient())
    # Should not raise
    await n.send("test")


@pytest.mark.asyncio
async def test_lark_notifier_handles_failure(monkeypatch) -> None:
    """LarkNotifier swallows errors from ABot."""
    n = LarkNotifier("https://fake-webhook.example.com")

    import lark_bots

    class FailBot:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def asend_text(self, text):
            raise ConnectionError("no network")

    monkeypatch.setattr(lark_bots, "ABot", FailBot)
    # Should not raise
    await n.send("test")


@pytest.mark.asyncio
async def test_multi_notifier_fans_out() -> None:
    """MultiNotifier sends to all backends."""
    messages: list[str] = []

    class Recorder(NullNotifier):
        def __init__(self, tag: str) -> None:
            self._tag = tag

        async def send(self, message: str) -> None:
            messages.append(f"{self._tag}:{message}")

    multi = MultiNotifier([Recorder("a"), Recorder("b")])
    await multi.send("hello")

    assert "a:hello" in messages
    assert "b:hello" in messages


@pytest.mark.asyncio
async def test_multi_notifier_tolerates_partial_failure() -> None:
    """One backend failing doesn't block others."""
    results: list[str] = []

    class FailNotifier(NullNotifier):
        async def send(self, message: str) -> None:
            raise RuntimeError("boom")

    class OkNotifier(NullNotifier):
        async def send(self, message: str) -> None:
            results.append(message)

    multi = MultiNotifier([FailNotifier(), OkNotifier()])
    await multi.send("test")

    assert results == ["test"]
