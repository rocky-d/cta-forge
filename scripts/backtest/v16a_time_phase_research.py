"""Research v16a time-phase sensitivity without touching live defaults.

This study is intentionally conservative and explanatory-first:
- We only test UTC-hour phase changes that the current 1h data can actually
  resolve. Minute-level offsets are out of scope without sub-hour bars.
- We sweep the 6h core sleeve phase (0..5 UTC-hour shifts) while keeping the
  1h overlay implementation unchanged.
- We also report a descriptive hour-of-day contribution view for the existing
  overlay sleeve, which helps audit whether the current UTC-hour soft prior is
  directionally supported.

Run:
    uv run python scripts/backtest/v16a_time_phase_research.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from data_service.store import ParquetStore
from executor.portfolio_backtest import (
    calculate_hourly_metrics,
    run_target_weight_backtest,
)
from executor.profiles.v16a_badscore_overlay import (
    DEFAULT_SYMBOLS,
    INITIAL_EQUITY,
    aggregate_phased_bars,
    align_weights,
    build_overlay_sleeve,
    expanding_badscore_gate,
    load_hourly_returns,
    normalize_gross,
    run_engine_positions,
    v10g_params,
)
from executor.signal_pipeline import (
    align_data,
    build_timeline,
    compute_signals,
    precompute,
)

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "backtest-results"

SPLITS = [
    ("2020_2021", "2020-01-01", "2021-12-31"),
    ("2022_2023", "2022-01-01", "2023-12-31"),
    ("2024_2026", "2024-01-01", "2026-12-31"),
]


@dataclass(frozen=True)
class MetricSnapshot:
    ret: float
    ann_return: float
    volatility: float
    sharpe: float
    max_dd: float
    ulcer: float


@dataclass(frozen=True)
class PhaseResult:
    phase: int
    start: str
    end: str
    bars: int
    total: MetricSnapshot
    splits: dict[str, MetricSnapshot]


@dataclass(frozen=True)
class OverlayHourResult:
    hour: int
    bars: int
    avg_ret_bp: float
    hit_rate: float
    total_return: float
    max_dd: float


@dataclass(frozen=True)
class WalkForwardResult:
    train_label: str
    test_label: str
    selected_phase: int
    train_sharpe: float
    test_sharpe: float
    test_return: float
    test_max_dd: float


def _metric_snapshot(metrics: dict[str, float | np.ndarray]) -> MetricSnapshot:
    return MetricSnapshot(
        ret=float(metrics["return"]),
        ann_return=float(metrics["ann_return"]),
        volatility=float(metrics["volatility"]),
        sharpe=float(metrics["sharpe"]),
        max_dd=float(metrics["max_dd"]),
        ulcer=float(metrics["ulcer"]),
    )


def build_v10g_sleeve_with_phase(
    *, phase_hours: int
) -> tuple[list[str], np.ndarray, list[tuple], list]:
    """Build the shifted v10g core sleeve using phased 1h aggregation."""
    store = ParquetStore(DATA_DIR)
    params = v10g_params()
    bars: dict[str, pl.DataFrame] = {}
    for symbol in DEFAULT_SYMBOLS:
        hourly = store.read(symbol, "1h")
        phased = aggregate_phased_bars(
            hourly,
            phase_hours=phase_hours,
            timeframe_hours=params.timeframe_hours,
        )
        if not phased.is_empty() and len(phased) >= 500:
            bars[symbol] = phased

    data = precompute(bars, params)
    timeline, ts_to_idx = build_timeline(bars)
    align_data(bars, data, ts_to_idx)
    signals = compute_signals(data, timeline, params, btc_filter=True)
    shifted = {
        symbol: np.concatenate([[0.0], np.asarray(signal[:-1], dtype=float)])
        for symbol, signal in signals.items()
    }
    syms, weights, curve, _trades = run_engine_positions(
        data, shifted, timeline, 200, params
    )
    return syms, weights, curve, timeline


def build_v16a_joint_with_core_phase(*, phase_hours: int) -> tuple[list, np.ndarray]:
    """Return v16a timeline + hourly return stream for a phased 6h core."""
    v_syms, v_weights, v_curve, v_timeline = build_v10g_sleeve_with_phase(
        phase_hours=phase_hours
    )
    o_syms, o_weights, o_curve, _o_trades, o_timeline = build_overlay_sleeve(
        backfill=False
    )

    if not v_curve or not o_curve:
        raise RuntimeError("insufficient history to build phased v16a target set")

    core_hours = v10g_params().timeframe_hours
    start = max(v_curve[0][0], o_curve[0][0])
    end = min(o_curve[-1][0], v_curve[-1][0] + timedelta(hours=core_hours * 2 - 1))
    timeline = [ts for ts in o_timeline if start <= ts <= end]
    symbols = sorted(set(v_syms) | set(o_syms))

    v_aligned = align_weights(
        v_timeline, v_syms, v_weights, timeline, symbols, forward_fill=True
    )
    o_aligned = align_weights(
        o_timeline, o_syms, o_weights, timeline, symbols, forward_fill=False
    )
    returns = load_hourly_returns(timeline, symbols)
    gate = expanding_badscore_gate(timeline)

    target = (0.5 * v_aligned + 0.5 * o_aligned) * gate[:, None]
    for i in range(target.shape[0]):
        capped = normalize_gross(
            {symbol: target[i, j] for j, symbol in enumerate(symbols)},
            gross_cap=1.0,
        )
        target[i] = np.array([capped.get(symbol, 0.0) for symbol in symbols])

    backtest = run_target_weight_backtest(
        timeline, returns, target, initial_equity=INITIAL_EQUITY
    )
    return timeline, backtest.returns


def build_phase_return_map() -> dict[int, tuple[list, np.ndarray]]:
    """Build per-phase timeline/return streams once for downstream analysis."""
    return {
        phase: build_v16a_joint_with_core_phase(phase_hours=phase) for phase in range(6)
    }


def run_phase_sweep(
    phase_returns: dict[int, tuple[list, np.ndarray]],
) -> list[PhaseResult]:
    """Evaluate UTC-hour phase choices for the 6h core sleeve."""
    results: list[PhaseResult] = []
    for phase in range(6):
        timeline, returns = phase_returns[phase]
        total = _metric_snapshot(
            calculate_hourly_metrics(returns, initial_equity=INITIAL_EQUITY)
        )
        days = np.array([ts.date().isoformat() for ts in timeline])
        split_metrics: dict[str, MetricSnapshot] = {}
        for name, lo, hi in SPLITS:
            mask = (days >= lo) & (days <= hi)
            split_metrics[name] = _metric_snapshot(
                calculate_hourly_metrics(returns[mask], initial_equity=INITIAL_EQUITY)
            )
        results.append(
            PhaseResult(
                phase=phase,
                start=timeline[0].isoformat(),
                end=timeline[-1].isoformat(),
                bars=len(timeline),
                total=total,
                splits=split_metrics,
            )
        )
    return results


def run_walkforward_selection(
    phase_returns: dict[int, tuple[list, np.ndarray]],
) -> list[WalkForwardResult]:
    """Run a small walk-forward phase-selection audit.

    This is not a production tuner. It is a sanity check for whether a phase
    preference learned on one regime meaningfully generalizes to the next.
    """

    windows = [
        (
            "2020_2021",
            "2020-01-01",
            "2021-12-31",
            "2022_2023",
            "2022-01-01",
            "2023-12-31",
        ),
        (
            "2020_2023",
            "2020-01-01",
            "2023-12-31",
            "2024_2026",
            "2024-01-01",
            "2026-12-31",
        ),
    ]
    out: list[WalkForwardResult] = []

    for train_label, train_lo, train_hi, test_label, test_lo, test_hi in windows:
        ranked: list[tuple[int, MetricSnapshot]] = []
        test_by_phase: dict[int, MetricSnapshot] = {}

        for phase, (timeline, returns) in phase_returns.items():
            days = np.array([ts.date().isoformat() for ts in timeline])
            train_mask = (days >= train_lo) & (days <= train_hi)
            test_mask = (days >= test_lo) & (days <= test_hi)
            train_metrics = _metric_snapshot(
                calculate_hourly_metrics(
                    returns[train_mask], initial_equity=INITIAL_EQUITY
                )
            )
            test_metrics = _metric_snapshot(
                calculate_hourly_metrics(
                    returns[test_mask], initial_equity=INITIAL_EQUITY
                )
            )
            ranked.append((phase, train_metrics))
            test_by_phase[phase] = test_metrics

        ranked.sort(key=lambda item: item[1].sharpe, reverse=True)
        selected_phase, train_metrics = ranked[0]
        test_metrics = test_by_phase[selected_phase]
        out.append(
            WalkForwardResult(
                train_label=train_label,
                test_label=test_label,
                selected_phase=selected_phase,
                train_sharpe=train_metrics.sharpe,
                test_sharpe=test_metrics.sharpe,
                test_return=test_metrics.ret,
                test_max_dd=test_metrics.max_dd,
            )
        )

    return out


def analyze_overlay_hour_profile() -> list[OverlayHourResult]:
    """Describe hour-of-day contribution for the current 1h overlay sleeve.

    This is descriptive only. It can falsify obviously bad priors, but it is not
    enough by itself to justify a production hour-selection rule.
    """
    _syms, _weights, curve, _trades, _timeline = build_overlay_sleeve(backfill=False)
    equity = np.array([eq for _ts, eq in curve], dtype=float)
    returns = np.zeros(len(curve), dtype=float)
    returns[1:] = equity[1:] / equity[:-1] - 1.0
    hours = np.array([ts.hour for ts, _eq in curve], dtype=int)
    results: list[OverlayHourResult] = []
    for hour in range(24):
        mask = hours == hour
        if mask.sum() < 20:
            continue
        sub = calculate_hourly_metrics(returns[mask], initial_equity=INITIAL_EQUITY)
        results.append(
            OverlayHourResult(
                hour=hour,
                bars=int(mask.sum()),
                avg_ret_bp=float(np.nanmean(returns[mask]) * 10_000),
                hit_rate=float((returns[mask] > 0).mean()),
                total_return=float(sub["return"]),
                max_dd=float(sub["max_dd"]),
            )
        )
    return sorted(results, key=lambda row: row.avg_ret_bp, reverse=True)


def render_phase_chart(
    phase_returns: dict[int, tuple[list, np.ndarray]],
    phase_results: list[PhaseResult],
    *,
    out_path: Path,
) -> None:
    """Render a compact comparison chart for the 6h phase sweep."""
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 10),
        gridspec_kw={"height_ratios": [2.2, 1.3, 1.2], "hspace": 0.28},
    )

    colors = {
        0: "#2ecc71",
        1: "#3498db",
        2: "#9b59b6",
        3: "#f39c12",
        4: "#e74c3c",
        5: "#7f8c8d",
    }

    for phase, (timeline, returns) in phase_returns.items():
        equity = INITIAL_EQUITY * np.cumprod(1.0 + returns)
        norm = equity / equity[0]
        dd = (np.maximum.accumulate(equity) - equity) / np.maximum.accumulate(equity)
        axes[0].plot(
            timeline,
            norm,
            label=f"phase {phase}",
            linewidth=1.7 if phase in {0, 1, 2} else 1.0,
            alpha=1.0 if phase in {0, 1, 2} else 0.65,
            color=colors[phase],
        )
        if phase in {0, 1, 2}:
            axes[1].plot(
                timeline,
                -dd * 100,
                linewidth=1.4,
                alpha=0.95,
                color=colors[phase],
                label=f"phase {phase}",
            )

    axes[0].set_title("v16a 6h core UTC phase sweep — normalized equity")
    axes[0].set_ylabel("Equity Index")
    axes[0].grid(True, alpha=0.18)
    axes[0].legend(ncol=3, fontsize=9, framealpha=0.85)

    axes[1].set_title("Top candidates drawdown (underwater plot)")
    axes[1].set_ylabel("DD %")
    axes[1].grid(True, alpha=0.18)
    axes[1].legend(fontsize=9, framealpha=0.85)

    ranking = sorted(phase_results, key=lambda row: row.total.sharpe, reverse=True)
    labels = [f"p{row.phase}" for row in ranking]
    sharpes = [row.total.sharpe for row in ranking]
    max_dds = [row.total.max_dd * 100 for row in ranking]
    x = np.arange(len(labels))
    axes[2].bar(x - 0.18, sharpes, width=0.36, color="#34495e", label="Sharpe")
    axes[2].bar(x + 0.18, max_dds, width=0.36, color="#c0392b", label="MaxDD %")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_title("Full-sample ranking summary")
    axes[2].grid(True, alpha=0.18, axis="y")
    axes[2].legend(fontsize=9, framealpha=0.85)

    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase_returns = build_phase_return_map()
    phase_results = run_phase_sweep(phase_returns)
    walkforward = run_walkforward_selection(phase_returns)
    overlay_hours = analyze_overlay_hour_profile()
    chart_path = OUT_DIR / "v16a_time_phase_research.png"
    render_phase_chart(phase_returns, phase_results, out_path=chart_path)

    payload = {
        "notes": {
            "scope": "UTC-hour phase sensitivity only; minute-level offsets need sub-hour data",
            "core_phase_sweep": "6h core rebuilt from 1h bars with phase 0..5",
            "overlay_hour_profile": "descriptive contribution audit for current 1h overlay hour prior",
            "warning": "use stability across splits and neighboring phases, not single best in-sample point",
        },
        "phase_results": [
            {
                "phase": row.phase,
                "start": row.start,
                "end": row.end,
                "bars": row.bars,
                "total": asdict(row.total),
                "splits": {
                    name: asdict(metrics) for name, metrics in row.splits.items()
                },
            }
            for row in phase_results
        ],
        "walkforward": [asdict(row) for row in walkforward],
        "overlay_hour_profile": [asdict(row) for row in overlay_hours],
    }
    out_path = OUT_DIR / "v16a_time_phase_research.json"
    out_path.write_text(json.dumps(payload, indent=2))

    print("== 6h core phase sweep (ranked by total Sharpe) ==")
    for row in sorted(phase_results, key=lambda item: item.total.sharpe, reverse=True):
        print(
            f"phase {row.phase}: sharpe={row.total.sharpe:.3f} "
            f"return={row.total.ret:.3f} maxdd={row.total.max_dd:.3f}"
        )

    print("\n== overlay hour profile (ranked by avg bp/hour) ==")
    for row in overlay_hours[:10]:
        print(
            f"hour {row.hour:02d}: avg_bp={row.avg_ret_bp:+.3f} "
            f"hit={row.hit_rate:.3f} maxdd={row.max_dd:.3f}"
        )

    print("\n== walk-forward phase selection ==")
    for row in walkforward:
        print(
            f"train {row.train_label} -> test {row.test_label}: "
            f"selected phase {row.selected_phase}, "
            f"test sharpe={row.test_sharpe:.3f}, "
            f"test return={row.test_return:.3f}, "
            f"test maxdd={row.test_max_dd:.3f}"
        )

    print(f"\nWrote {out_path}")
    print(f"Wrote {chart_path}")


if __name__ == "__main__":
    main()
