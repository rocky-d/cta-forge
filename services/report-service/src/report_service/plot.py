"""Plotting utilities for backtest results."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from datetime import datetime

matplotlib.use("Agg")


def plot_backtest(
    equity_curve: list[tuple[datetime, float]],
    btc_prices: list[tuple[datetime, float]] | None = None,
    eth_prices: list[tuple[datetime, float]] | None = None,
    metrics: dict[str, Any] | None = None,
    yearly: dict[str, float] | None = None,
    title_extra: str = "",
    initial_equity: float = 10_000.0,
    dpi: int = 200,
    strategy_label: str = "CTA-Forge v10g",
) -> bytes:
    """Generate three-panel backtest chart as PNG bytes.

    Panels:
        1. Equity curve with BTC/ETH indexed price overlay
        2. Drawdown percentage
        3. Monthly returns bar chart

    Args:
        equity_curve: [(datetime, equity), ...]
        btc_prices: [(datetime, close), ...] for overlay
        eth_prices: [(datetime, close), ...] for overlay
        metrics: dict with sharpe_ratio, total_return, etc.
        yearly: {year_str: return_pct, ...}
        title_extra: additional text appended to chart title
        initial_equity: starting equity for baseline reference
        dpi: output image DPI

    Returns:
        PNG image as bytes.
    """
    if not equity_curve:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title("No data")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    ts_b = [e[0] for e in equity_curve]
    eq_b = [e[1] for e in equity_curve]
    curve_start_ts = ts_b[0]

    btc_prices = btc_prices or []
    eth_prices = eth_prices or []

    # Filter overlay prices to equity curve time range
    btc_filt = [(t, p) for t, p in btc_prices if t >= curve_start_ts]
    eth_filt = [(t, p) for t, p in eth_prices if t >= curve_start_ts]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(24, 12),
        height_ratios=[3.5, 1, 1.5],
        gridspec_kw={"hspace": 0.15},
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.06, right=0.94, top=0.95, hspace=0.15)

    # === Top panel: Equity (left Y) + BTC/ETH indexed price (right Y) ===
    ax = axes[0]
    ax2 = ax.twinx()

    ln_btc = []
    ln_eth = []
    if btc_filt:
        btc_base = btc_filt[0][1]
        ln_btc = ax2.plot(
            [p[0] for p in btc_filt],
            [p[1] / btc_base * 100 for p in btc_filt],
            linewidth=0.6,
            color="#F7931A",
            alpha=0.4,
            label="BTC (indexed)",
        )
    if eth_filt:
        eth_base = eth_filt[0][1]
        ln_eth = ax2.plot(
            [p[0] for p in eth_filt],
            [p[1] / eth_base * 100 for p in eth_filt],
            linewidth=0.6,
            color="#627EEA",
            alpha=0.4,
            label="ETH (indexed)",
        )
    ax2.set_ylabel("BTC / ETH Index (start = 100)", fontsize=9, color="#888888")
    ax2.tick_params(axis="y", labelcolor="#888888", labelsize=8)

    ln_eq = ax.plot(
        ts_b,
        eq_b,
        linewidth=1.8,
        color="#2ecc71",
        label=strategy_label,
        zorder=10,
    )
    ax.axhline(
        y=initial_equity, color="#7f8c8d", linestyle="--", linewidth=0.5, alpha=0.4
    )
    ax.fill_between(
        ts_b,
        initial_equity,
        eq_b,
        where=[e >= initial_equity for e in eq_b],
        alpha=0.1,
        color="#2ecc71",
        zorder=5,
    )
    ax.fill_between(
        ts_b,
        initial_equity,
        eq_b,
        where=[e < initial_equity for e in eq_b],
        alpha=0.1,
        color="#e74c3c",
        zorder=5,
    )

    # Keep equity curve on top of overlay lines
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    title = "CTA-Forge v10g — Backtest (V10GDecisionEngine)"
    if title_extra:
        title += f"\n{title_extra}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Equity ($)", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _p: f"${x:,.0f}"))

    lns = ln_eq + ln_btc + ln_eth
    labs = [line.get_label() for line in lns]
    ax.legend(lns, labs, loc="upper left", fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.15)

    # Bottom xlabel: metrics + BTC/ETH B&H
    label_parts = []
    if metrics:
        m = metrics
        metric_parts = [
            f"Return: {m.get('total_return', 0) * 100:+.1f}%",
            f"Ann: {m.get('annualized_return', 0) * 100:+.1f}%",
            f"Sharpe: {m.get('sharpe_ratio', 0):.2f}",
            f"Sortino: {m.get('sortino_ratio', 0):.2f}",
            f"MaxDD: {m.get('max_drawdown', 0) * 100:.1f}%",
            f"Calmar: {m.get('calmar_ratio', 0):.2f}",
        ]
        if m.get("profit_factor") is not None:
            metric_parts.append(f"PF: {m.get('profit_factor', 0):.2f}")
        if m.get("win_rate") is not None:
            metric_parts.append(f"Win: {m.get('win_rate', 0) * 100:.1f}%")
        if m.get("num_trades") is not None:
            metric_parts.append(f"Trades: {m.get('num_trades', 0)}")
        if m.get("ulcer_index") is not None:
            metric_parts.append(f"Ulcer: {m.get('ulcer_index', 0):.4f}")
        label_parts.append("   ".join(metric_parts))
    if btc_filt:
        btc_ret = (btc_filt[-1][1] / btc_filt[0][1] - 1) * 100
        label_parts.append(f"BTC B&H: {btc_ret:+.0f}%")
    if eth_filt:
        eth_ret = (eth_filt[-1][1] / eth_filt[0][1] - 1) * 100
        label_parts.append(f"ETH B&H: {eth_ret:+.0f}%")
    if yearly:
        yr_str = "  ".join(f"{yr}: {ret:+.1f}%" for yr, ret in sorted(yearly.items()))
        label_parts.append(yr_str)
    if label_parts:
        # Join metrics on one line, B&H and yearly on next
        main_line = label_parts[0] if label_parts else ""
        extra = "   |   ".join(label_parts[1:]) if len(label_parts) > 1 else ""
        full_label = f"{main_line}   |   {extra}" if extra else main_line
        axes[2].set_xlabel(
            full_label, fontsize=7.5, family="monospace", color="#2c3e50", labelpad=6
        )

    # === Middle: Drawdown ===
    eq_arr = np.array(eq_b)
    rm = np.maximum.accumulate(eq_arr)
    dd_pct = (rm - eq_arr) / rm * 100
    axes[1].fill_between(ts_b, 0, -dd_pct, color="#e74c3c", alpha=0.5)
    axes[1].set_ylabel("DD %")
    axes[1].grid(True, alpha=0.15)

    # === Bottom: Monthly returns ===
    monthly: dict[str, dict[str, float]] = {}
    for ii in range(1, len(eq_b)):
        key = ts_b[ii].strftime("%Y-%m")
        if key not in monthly:
            monthly[key] = {"start": eq_b[ii - 1]}
        monthly[key]["end"] = eq_b[ii]
    mos = list(monthly.keys())
    rets = [
        (monthly[m_]["end"] - monthly[m_]["start"]) / monthly[m_]["start"] * 100
        for m_ in mos
    ]
    axes[2].bar(
        range(len(mos)),
        rets,
        color=["#2ecc71" if r >= 0 else "#e74c3c" for r in rets],
        alpha=0.8,
        width=0.8,
    )
    step = max(1, len(mos) // 24)
    axes[2].set_xticks(range(0, len(mos), step))
    axes[2].set_xticklabels(
        [mos[ii] for ii in range(0, len(mos), step)],
        rotation=45,
        ha="right",
        fontsize=6,
    )
    axes[2].set_ylabel("Monthly %")
    axes[2].axhline(y=0, color="black", linewidth=0.5)
    axes[2].grid(True, alpha=0.15, axis="y")

    pos_months = sum(1 for r in rets if r > 0)
    if mos:
        axes[2].text(
            0.02,
            0.95,
            f"Positive: {pos_months}/{len(rets)} months "
            f"({pos_months / len(rets) * 100:.0f}%)",
            transform=axes[2].transAxes,
            fontsize=8,
            va="top",
        )

    for a in axes[:2]:
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        a.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Legacy simple charts (kept for backward compatibility) ───────


def plot_equity_curve(
    equity_curve: list[tuple[datetime, float]],
    title: str = "Equity Curve",
) -> bytes:
    """Generate simple equity curve plot as PNG bytes."""
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
    ax.plot(timestamps, equities, linewidth=1.5, color="#2ecc71")
    ax.fill_between(timestamps, equities, alpha=0.2, color="#2ecc71")

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
    ax.fill_between(timestamps, 0, -drawdowns, color="#e74c3c", alpha=0.6)
    ax.plot(timestamps, -drawdowns, color="#c0392b", linewidth=1)

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
