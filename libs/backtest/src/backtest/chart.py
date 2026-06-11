"""Research chart generation for cta-forge backtest comparisons.

Convention: 16:9 PNG with three vertically stacked panels:
  1. Normalized equity (largest)
  2. Underwater drawdown
  3. Monthly P&L bar chart

At most 4 configs per chart. The module consumes ``ChartSeries`` objects
and knows nothing about strategies, profiles, or data sources.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from .result import ChartSeries

DEFAULT_COLORS = ["#2f81f7", "#f0883e", "#3fb950", "#f85149"]

DEFAULT_FIGSIZE = (20, 11.25)  # 16:9
DEFAULT_DPI = 150
MAX_CONFIGS = 4


# ── Panel specification ──────────────────────────────────────────


@dataclass
class PanelSpec:
    """Definition of one panel in a comparison chart."""

    kind: str  # "equity", "drawdown", "monthly_bar"
    height_ratio: float  # relative height (e.g. 6, 1.5, 2.5)
    title: str = ""
    ylabel: str = ""


DEFAULT_PANELS = [
    PanelSpec(kind="equity", height_ratio=6, ylabel="Equity Index"),
    PanelSpec(kind="drawdown", height_ratio=1.5, ylabel="Drawdown %"),
    PanelSpec(kind="monthly_bar", height_ratio=2.5, ylabel="Monthly Return %"),
]

LEGEND_FONTSIZE = 7.5
LABEL_FONTSIZE = 9
TICK_FONTSIZE = 7.5
GRID_ALPHA = 0.25


# ── Chart creation ───────────────────────────────────────────────


def create_comparison_figure(
    results: list[ChartSeries],
    *,
    title: str = "",
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    panels: list[PanelSpec] | None = None,
    colors: list[str] | None = None,
) -> plt.Figure:
    """Create a multi-panel comparison chart.

    Args:
        results: One ChartSeries per config (max 4).
        title: Overall chart title (panel 1 subtitle).
        figsize: Figure size in inches (default 16:9).
        panels: Panel definitions (default: 3-panel convention).
        colors: Colors per series (default: DEFAULT_COLORS).

    Returns:
        matplotlib Figure ready for save or display.
    """
    if len(results) > MAX_CONFIGS:
        warnings.warn(
            f"Chart convention: max {MAX_CONFIGS} configs per chart, got "
            f"{len(results)}. The chart will be generated but may be cluttered.",
            stacklevel=2,
        )

    panels = panels or DEFAULT_PANELS
    colors = colors or DEFAULT_COLORS
    n_panels = len(panels)

    fig = plt.figure(figsize=figsize, dpi=DEFAULT_DPI)
    hspace = 0.35 if n_panels >= 3 else 0.25
    gs = fig.add_gridspec(
        n_panels, 1, height_ratios=[p.height_ratio for p in panels], hspace=hspace
    )

    for pi, panel in enumerate(panels):
        ax = fig.add_subplot(gs[pi])
        if panel.kind == "equity":
            _draw_equity_panel(ax, results, colors, title)
        elif panel.kind == "drawdown":
            _draw_drawdown_panel(ax, results, colors)
        elif panel.kind == "monthly_bar":
            _draw_monthly_bar_panel(ax, results, colors)
        else:
            raise ValueError(f"Unknown panel kind: {panel.kind!r}")

    return fig


def save_figure(fig: plt.Figure, path: str | Path, *, dpi: int = DEFAULT_DPI) -> Path:
    """Save figure to PNG and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white", format="png")
    plt.close(fig)
    return path


# ── Panel renderers ──────────────────────────────────────────────


def _draw_equity_panel(
    ax: plt.Axes, results: list[ChartSeries], colors: list[str], title: str
) -> None:
    for i, r in enumerate(results):
        c = colors[i % len(colors)]
        t = r.timestamps if r.timestamps is not None else np.arange(len(r.equity))
        lbl = r.label
        if r.metrics is not None:
            m = r.metrics
            lbl += (
                f"  |  Sharpe {m.sharpe_ratio:.2f}  "
                f"Return {m.total_return * 100:.1f}%  "
                f"Ann. {m.annualized_return * 100:.1f}%  "
                f"MaxDD {m.max_drawdown * 100:.1f}%"
            )
        ax.plot(t, r.equity, color=c, linewidth=1.0, label=lbl, alpha=0.92)

    ax.axhline(y=1.0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_ylabel("Equity Index", fontsize=LABEL_FONTSIZE)
    if title:
        ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper left", framealpha=0.85)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=GRID_ALPHA)


def _draw_drawdown_panel(
    ax: plt.Axes, results: list[ChartSeries], colors: list[str]
) -> None:
    for i, r in enumerate(results):
        c = colors[i % len(colors)]
        t = r.timestamps if r.timestamps is not None else np.arange(len(r.drawdown))
        dd_pct = r.drawdown * 100  # convert fraction → %
        ax.fill_between(t, 0, dd_pct, color=c, alpha=0.30, linewidth=0)
        ax.plot(t, dd_pct, color=c, linewidth=0.7, alpha=0.85)

    ax.set_ylabel("Drawdown %", fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))


def _draw_monthly_bar_panel(
    ax: plt.Axes, results: list[ChartSeries], colors: list[str]
) -> None:
    all_months = sorted(set().union(*(set(r.monthly_returns) for r in results)))
    if not all_months:
        ax.text(
            0.5,
            0.5,
            "No monthly data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=LABEL_FONTSIZE,
        )
        return

    n = len(all_months)
    n_runs = len(results)
    bar_w = 0.8 / n_runs
    x = np.arange(n)

    for ri, r in enumerate(results):
        c = colors[ri % len(colors)]
        vals = [r.monthly_returns.get(m, 0.0) * 100 for m in all_months]  # fraction→%
        offset = (ri - (n_runs - 1) / 2) * bar_w
        ax.bar(
            x + offset,
            vals,
            bar_w,
            color=c,
            alpha=0.82,
            label=r.label,
            linewidth=0,
        )

    # Smart x-tick labeling: show ~6 ticks, include year boundaries
    tick_positions = []
    tick_labels = []
    for i, m in enumerate(all_months):
        month = m[5:7]
        if i % max(1, n // 6) == 0 or month == "01":
            tick_positions.append(i)
            tick_labels.append(m)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=6.5)

    ax.set_ylabel("Monthly Return %", fontsize=LABEL_FONTSIZE)
    if len(results) > 1:
        ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper left", framealpha=0.85)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(True, axis="y", alpha=GRID_ALPHA)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
