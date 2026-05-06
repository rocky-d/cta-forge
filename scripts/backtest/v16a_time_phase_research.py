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


def _metric_snapshot(metrics: dict[str, float | np.ndarray]) -> MetricSnapshot:
    return MetricSnapshot(
        ret=float(metrics["return"]),
        ann_return=float(metrics["ann_return"]),
        volatility=float(metrics["volatility"]),
        sharpe=float(metrics["sharpe"]),
        max_dd=float(metrics["max_dd"]),
        ulcer=float(metrics["ulcer"]),
    )


def aggregate_phased_bars(
    df: pl.DataFrame, *, phase_hours: int, timeframe_hours: int
) -> pl.DataFrame:
    """Aggregate 1h bars into a synthetic phased timeframe.

    A 6h phase of 1 means bars start at 01:00, 07:00, 13:00, 19:00 UTC, etc.
    Only full windows are kept.
    """
    if df.is_empty():
        return df
    ts = df["open_time"].to_list()
    epoch_hours = np.array([int(t.timestamp() // 3600) for t in ts], dtype=np.int64)
    grp = (epoch_hours - phase_hours) // timeframe_hours
    phase_mod = (epoch_hours - phase_hours) % timeframe_hours
    joined = df.with_columns(pl.Series("_grp", grp), pl.Series("_phase_mod", phase_mod))
    return (
        joined.group_by("_grp")
        .agg(
            pl.col("open_time").first().alias("open_time"),
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("quote_volume").sum().alias("quote_volume"),
            pl.col("_phase_mod").first().alias("_phase_mod"),
            pl.len().alias("_n"),
        )
        .filter((pl.col("_phase_mod") == 0) & (pl.col("_n") == timeframe_hours))
        .drop("_phase_mod", "_n")
        .sort("open_time")
    )


def build_v10g_sleeve_with_phase(
    *, phase_hours: int
) -> tuple[list[str], np.ndarray, list[tuple], list]:
    """Build the shifted v10g core sleeve using a synthetic 6h phase."""
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
    syms, weights, curve, trades = run_engine_positions(
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


def run_phase_sweep() -> list[PhaseResult]:
    """Evaluate UTC-hour phase choices for the 6h core sleeve."""
    results: list[PhaseResult] = []
    for phase in range(6):
        timeline, returns = build_v16a_joint_with_core_phase(phase_hours=phase)
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase_results = run_phase_sweep()
    overlay_hours = analyze_overlay_hour_profile()

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

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
