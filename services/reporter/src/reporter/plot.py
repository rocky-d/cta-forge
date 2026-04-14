"""Plotting utilities for backtest results."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from datetime import datetime

matplotlib.use("Agg")


def plot_equity_curve(
    equity_curve: list[tuple[datetime, float]],
    title: str = "Equity Curve",
) -> bytes:
    """Generate equity curve plot as PNG bytes.

    Returns PNG image bytes that can be saved or sent via HTTP.
    """
    if not equity_curve:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title("No data")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    timestamps = [e[0] for e in equity_curve]
    equities = [e[1] for e in equity_curve]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(timestamps, equities, linewidth=1.5, color="#2ecc71")  # type: ignore[arg-type]
    ax.fill_between(timestamps, equities, alpha=0.2, color="#2ecc71")  # type: ignore[arg-type]

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    fig.autofmt_xdate()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_drawdown(equity_curve: list[tuple[datetime, float]]) -> bytes:
    """Generate drawdown chart as PNG bytes."""
    if not equity_curve:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_title("No data")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    timestamps = [e[0] for e in equity_curve]
    equities = np.array([e[1] for e in equity_curve])

    running_max = np.maximum.accumulate(equities)
    drawdowns = (running_max - equities) / running_max * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(timestamps, 0, -drawdowns, color="#e74c3c", alpha=0.6)  # type: ignore[arg-type]
    ax.plot(timestamps, -drawdowns, color="#c0392b", linewidth=1)  # type: ignore[arg-type]

    ax.set_title("Drawdown", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown %")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=min(-drawdowns.max() * 1.1, -1))

    fig.autofmt_xdate()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_returns_distribution(trades: list[dict]) -> bytes:
    """Generate trade returns distribution histogram as PNG bytes."""
    pnls = [t.get("pnl", 0.0) for t in trades if "pnl" in t]

    fig, ax = plt.subplots(figsize=(10, 5))

    if not pnls:
        ax.set_title("No trades")
    else:
        ax.hist(pnls, bins=50, color="#3498db", alpha=0.7, edgecolor="white")
        ax.axvline(x=0, color="#e74c3c", linestyle="--", linewidth=2)
        ax.axvline(
            x=np.mean(pnls),
            color="#2ecc71",
            linestyle="-",
            linewidth=2,
            label=f"Mean: ${np.mean(pnls):.2f}",
        )
        ax.set_title("Trade P&L Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("P&L ($)")
        ax.set_ylabel("Frequency")
        ax.legend()

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
