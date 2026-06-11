"""Research chart generation for cta-forge backtest comparisons.

Convention: 16:9 PNG with three vertically stacked panels:
  1. Normalized equity (largest)
  2. Underwater drawdown (filled downward from zero)
  3. Monthly P&L chart (bar or line)

All panels share a common datetime x-axis.  Month-level grid lines are
visible on all panels; month labels appear only on the bottom panel.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from .result import ChartSeries

# ── Colour constants ─────────────────────────────────────────────

EQUITY_COLORS = ["#2f81f7", "#f0883e", "#3fb950", "#f85149"]
DD_COLOR = "#f85149"
BAR_POSITIVE_COLOR = "#3fb950"
BAR_NEGATIVE_COLOR = "#f85149"

DEFAULT_FIGSIZE = (20, 11.25)  # 16:9
DEFAULT_DPI = 150
MAX_CONFIGS = 4

# ── Panel specification ──────────────────────────────────────────


@dataclass
class PanelSpec:
    kind: str  # "equity" | "drawdown" | "monthly_bar" | "monthly_line"
    height_ratio: float
    title: str = ""
    ylabel: str = ""


DEFAULT_PANELS = [
    PanelSpec(kind="equity", height_ratio=6, ylabel="Equity Index"),
    PanelSpec(kind="drawdown", height_ratio=1.5, ylabel="Drawdown %"),
    PanelSpec(kind="monthly_bar", height_ratio=2.5, ylabel="Monthly Return %"),
]

LEGEND_FONTSIZE = 7.5
LABEL_FONTSIZE = 9
TICK_FONTSIZE = 6.5
GRID_ALPHA = 0.20
MONTH_LABEL_ROTATION = 45


# ── Chart creation ───────────────────────────────────────────────


def create_comparison_figure(
    results: list[ChartSeries],
    *,
    title: str = "",
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    panels: list[PanelSpec] | None = None,
    equity_colors: list[str] | None = None,
    drawdown_colors: list[str] | None = None,
    monthly_colors: list[str] | None = None,
) -> plt.Figure:
    if len(results) > MAX_CONFIGS:
        warnings.warn(
            f"Chart convention: max {MAX_CONFIGS} configs per chart, got "
            f"{len(results)}. The chart will be generated but may be cluttered.",
            stacklevel=2,
        )

    panels = panels or DEFAULT_PANELS
    ec = equity_colors or [
        r.color or EQUITY_COLORS[i % len(EQUITY_COLORS)] for i, r in enumerate(results)
    ]
    n = len(panels)

    fig = plt.figure(figsize=figsize, dpi=DEFAULT_DPI)
    gs = fig.add_gridspec(
        n,
        1,
        height_ratios=[p.height_ratio for p in panels],
        hspace=0.08 if n >= 3 else 0.15,
    )

    axes: list[plt.Axes] = []
    for pi, panel in enumerate(panels):
        sharex = axes[0] if pi > 0 else None
        ax = fig.add_subplot(gs[pi], sharex=sharex)
        axes.append(ax)

        if panel.kind == "equity":
            _draw_equity_panel(ax, results, ec, title)
        elif panel.kind == "drawdown":
            _draw_drawdown_panel(ax, results, drawdown_colors or ec)
        elif panel.kind == "monthly_bar":
            _draw_monthly_bar_panel(ax, results, monthly_colors)
        elif panel.kind == "monthly_line":
            _draw_monthly_line_panel(ax, results, monthly_colors or ec)
        else:
            raise ValueError(f"Unknown panel kind: {panel.kind!r}")

    # ── Shared x-axis formatting ─────────────────────────────────
    _apply_shared_x_axis(axes, len(panels))

    return fig


def _apply_shared_x_axis(axes: list[plt.Axes], n_panels: int) -> None:
    """Month-level grid lines on all panels; labels only on the bottom."""
    bottom = axes[-1]
    # Month locator: every single month
    bottom.xaxis.set_major_locator(mdates.MonthLocator())
    bottom.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(bottom.xaxis.get_major_locator())
    )
    bottom.tick_params(axis="x", labelsize=TICK_FONTSIZE, rotation=MONTH_LABEL_ROTATION)

    for i, ax in enumerate(axes):
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.tick_params(axis="x", which="major", labelsize=TICK_FONTSIZE)
        ax.grid(True, which="major", axis="x", alpha=GRID_ALPHA, linewidth=0.5)
        if i < n_panels - 1:
            # Hide labels on upper panels
            ax.tick_params(axis="x", labelbottom=False)


def save_figure(fig: plt.Figure, path: str | Path, *, dpi: int = DEFAULT_DPI) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white", format="png")
    plt.close(fig)
    return path


# ── Panel renderers ──────────────────────────────────────────────


def _draw_equity_panel(
    ax: plt.Axes,
    results: list[ChartSeries],
    colors: list[str],
    title: str,
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
    ax: plt.Axes,
    results: list[ChartSeries],
    colors: list[str],
) -> None:
    for i, r in enumerate(results):
        c = colors[i % len(colors)]
        t = r.timestamps if r.timestamps is not None else np.arange(len(r.drawdown))
        dd_neg = -(r.drawdown * 100)
        ax.fill_between(t, 0, dd_neg, color=c, alpha=0.30, linewidth=0)
        ax.plot(t, dd_neg, color=c, linewidth=0.7, alpha=0.85)

    ax.set_ylabel("Drawdown %", fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))


def _draw_monthly_bar_panel(
    ax: plt.Axes,
    results: list[ChartSeries],
    colors: list[str] | None,
) -> None:
    """Monthly P&L bar chart on a datetime x-axis shared with upper panels."""
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

    # Use datetime positions so grid lines align with panels 1-2
    import datetime as dt

    x_dates = [dt.datetime(int(m[:4]), int(m[5:7]), 1) for m in all_months]
    n_runs = len(results)
    bar_w_days = 20.0 / n_runs

    for ri, r in enumerate(results):
        vals = [r.monthly_returns.get(m, 0.0) * 100 for m in all_months]
        offset_days = (ri - (n_runs - 1) / 2) * bar_w_days
        x_pos = [d + dt.timedelta(days=offset_days) for d in x_dates]

        if colors:
            c = colors[ri % len(colors)]
            ax.bar(
                x_pos,
                vals,
                dt.timedelta(days=bar_w_days * 0.9),
                color=c,
                alpha=0.82,
                label=r.label if n_runs > 1 else None,
                linewidth=0,
            )
        else:
            pos_vals = [max(v, 0) for v in vals]
            neg_vals = [min(v, 0) for v in vals]
            ax.bar(
                x_pos,
                pos_vals,
                dt.timedelta(days=bar_w_days * 0.9),
                color=BAR_POSITIVE_COLOR,
                alpha=0.82,
                linewidth=0,
            )
            ax.bar(
                x_pos,
                neg_vals,
                dt.timedelta(days=bar_w_days * 0.9),
                color=BAR_NEGATIVE_COLOR,
                alpha=0.82,
                linewidth=0,
            )
            if n_runs > 1:
                from matplotlib.patches import Patch

                if ri == 0:
                    ax.legend(
                        handles=[
                            Patch(color=BAR_POSITIVE_COLOR, label="+"),
                            Patch(color=BAR_NEGATIVE_COLOR, label="−"),
                        ],
                        fontsize=LEGEND_FONTSIZE,
                        loc="upper left",
                        framealpha=0.85,
                    )

    ax.set_ylabel("Monthly Return %", fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(True, axis="y", alpha=GRID_ALPHA)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))


def _draw_monthly_line_panel(
    ax: plt.Axes,
    results: list[ChartSeries],
    colors: list[str],
) -> None:
    """Monthly P&L line chart on a datetime x-axis."""
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

    import datetime as dt

    x_dates = [dt.datetime(int(m[:4]), int(m[5:7]), 1) for m in all_months]
    for ri, r in enumerate(results):
        c = colors[ri % len(colors)]
        vals = [r.monthly_returns.get(m, 0.0) * 100 for m in all_months]
        ax.plot(
            x_dates,
            vals,
            color=c,
            linewidth=1.0,
            alpha=0.88,
            marker="o",
            markersize=2,
            markevery=1,
            label=r.label,
        )

    ax.set_ylabel("Monthly Return %", fontsize=LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper left", framealpha=0.85)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
