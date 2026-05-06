"""Extended robustness checks for v16a 6h core phase research.

This script intentionally builds on ``v16a_time_phase_research.py`` instead of
creating another strategy implementation. It answers narrower follow-up
questions:
- Does the leading phase survive rolling out-of-sample selection?
- Is the phase edge smooth around neighboring phases or an isolated spike?
- How do phase candidates behave in high-volatility / high-cohesion regimes?
- How sensitive are candidates to simple live-like execution thresholds?

Run:
    uv run python scripts/backtest/v16a_phase_robustness_research.py
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from executor.portfolio_backtest import (  # noqa: E402
    calculate_hourly_metrics,
    run_execution_backtest,
    run_target_weight_backtest,
)
from executor.profiles.v16a_badscore_overlay import (  # noqa: E402
    INITIAL_EQUITY,
    align_weights,
    build_overlay_sleeve,
    expanding_badscore_gate,
    load_hourly_returns,
    normalize_gross,
    v10g_params,
)
from scripts.backtest.v16a_time_phase_research import (  # noqa: E402
    build_v10g_sleeve_with_phase,
)

OUT_DIR = ROOT / "backtest-results"

ROLLING_WINDOWS = [
    ("2020_2021", "2020-01-01", "2021-12-31", "2022", "2022-01-01", "2022-12-31"),
    ("2021_2022", "2021-01-01", "2022-12-31", "2023", "2023-01-01", "2023-12-31"),
    ("2022_2023", "2022-01-01", "2023-12-31", "2024", "2024-01-01", "2024-12-31"),
    ("2023_2024", "2023-01-01", "2024-12-31", "2025", "2025-01-01", "2025-12-31"),
    ("2024_2025", "2024-01-01", "2025-12-31", "2026_ytd", "2026-01-01", "2026-12-31"),
]

EXPANDING_WINDOWS = [
    ("2020_2021", "2020-01-01", "2021-12-31", "2022_2023", "2022-01-01", "2023-12-31"),
    ("2020_2022", "2020-01-01", "2022-12-31", "2023", "2023-01-01", "2023-12-31"),
    ("2020_2023", "2020-01-01", "2023-12-31", "2024_2026", "2024-01-01", "2026-12-31"),
    ("2020_2024", "2020-01-01", "2024-12-31", "2025_2026", "2025-01-01", "2026-12-31"),
]

EXECUTION_MIN_NOTIONALS = [0.0, 5.0, 10.0, 25.0, 50.0, 100.0]
EPS = 1e-12


@dataclass(frozen=True)
class MetricSnapshot:
    ret: float
    ann_return: float
    volatility: float
    sharpe: float
    max_dd: float
    ulcer: float


@dataclass(frozen=True)
class PhaseTargetSet:
    phase: int
    timeline: list[datetime]
    symbols: list[str]
    returns: np.ndarray
    target_weights: np.ndarray
    portfolio_returns: np.ndarray
    turnover: np.ndarray


@dataclass(frozen=True)
class PhaseSummary:
    phase: int
    start: str
    end: str
    bars: int
    metrics: MetricSnapshot
    avg_gross: float
    avg_turnover_per_hour: float
    active_hours: int
    order_events: int


@dataclass(frozen=True)
class SelectionWindowResult:
    mode: str
    train_label: str
    test_label: str
    selected_phase: int
    best_test_phase: int
    train_sharpe: float
    test_sharpe: float
    test_return: float
    test_max_dd: float
    selected_test_rank: int


@dataclass(frozen=True)
class RegimeResult:
    regime: str
    phase: int
    bars: int
    metrics: MetricSnapshot


@dataclass(frozen=True)
class ExecutionSensitivityResult:
    phase: int
    min_order_notional: float
    metrics: MetricSnapshot
    avg_turnover_per_hour: float
    avg_orders_per_hour: float
    ignored_turnover_per_hour: float
    ignored_gross_avg: float


def metric_snapshot(returns: np.ndarray) -> MetricSnapshot:
    metrics = calculate_hourly_metrics(returns, initial_equity=INITIAL_EQUITY)
    return MetricSnapshot(
        ret=float(metrics["return"]),
        ann_return=float(metrics["ann_return"]),
        volatility=float(metrics["volatility"]),
        sharpe=float(metrics["sharpe"]),
        max_dd=float(metrics["max_dd"]),
        ulcer=float(metrics["ulcer"]),
    )


def date_mask(timeline: list[datetime], lo: str, hi: str) -> np.ndarray:
    days = np.array([ts.date().isoformat() for ts in timeline])
    return (days >= lo) & (days <= hi)


def rolling_mean_prev(x: np.ndarray, window: int) -> np.ndarray:
    valid = np.isfinite(x)
    vals = np.where(valid, x, 0.0)
    cs = np.concatenate([[0.0], np.cumsum(vals)])
    cc = np.concatenate([[0], np.cumsum(valid.astype(int))])
    out = np.full(len(x), np.nan)
    for i in range(window, len(x)):
        n = cc[i] - cc[i - window]
        if n > 0:
            out[i] = (cs[i] - cs[i - window]) / n
    return out


def rolling_std_prev(x: np.ndarray, window: int) -> np.ndarray:
    mean = rolling_mean_prev(x, window)
    mean2 = rolling_mean_prev(x * x, window)
    return np.sqrt(np.maximum(mean2 - mean * mean, 0.0))


def build_joint_target_with_core_phase(*, phase_hours: int) -> PhaseTargetSet:
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
            {symbol: target[i, j] for j, symbol in enumerate(symbols)}, gross_cap=1.0
        )
        target[i] = np.array([capped.get(symbol, 0.0) for symbol in symbols])

    backtest = run_target_weight_backtest(
        timeline, returns, target, initial_equity=INITIAL_EQUITY
    )
    return PhaseTargetSet(
        phase=phase_hours,
        timeline=timeline,
        symbols=symbols,
        returns=returns,
        target_weights=target,
        portfolio_returns=backtest.returns,
        turnover=backtest.turnover,
    )


def summarize_phase(phase_set: PhaseTargetSet) -> PhaseSummary:
    gross = np.sum(np.abs(phase_set.target_weights), axis=1)
    order_events = int(np.sum(np.abs(np.diff(phase_set.target_weights, axis=0)) > EPS))
    return PhaseSummary(
        phase=phase_set.phase,
        start=phase_set.timeline[0].isoformat(),
        end=phase_set.timeline[-1].isoformat(),
        bars=len(phase_set.timeline),
        metrics=metric_snapshot(phase_set.portfolio_returns),
        avg_gross=float(np.mean(gross)),
        avg_turnover_per_hour=float(np.mean(phase_set.turnover)),
        active_hours=int(np.sum(gross > EPS)),
        order_events=order_events,
    )


def run_selection_windows(
    phase_sets: dict[int, PhaseTargetSet],
    windows: list[tuple[str, str, str, str, str, str]],
    *,
    mode: str,
) -> list[SelectionWindowResult]:
    out: list[SelectionWindowResult] = []
    for train_label, train_lo, train_hi, test_label, test_lo, test_hi in windows:
        train_ranked: list[tuple[int, MetricSnapshot]] = []
        test_metrics: dict[int, MetricSnapshot] = {}
        for phase, phase_set in phase_sets.items():
            train = date_mask(phase_set.timeline, train_lo, train_hi)
            test = date_mask(phase_set.timeline, test_lo, test_hi)
            train_ranked.append(
                (phase, metric_snapshot(phase_set.portfolio_returns[train]))
            )
            test_metrics[phase] = metric_snapshot(phase_set.portfolio_returns[test])

        train_ranked.sort(key=lambda item: item[1].sharpe, reverse=True)
        selected_phase, train_metrics = train_ranked[0]
        test_ranked = sorted(
            test_metrics.items(), key=lambda item: item[1].sharpe, reverse=True
        )
        test_rank = [phase for phase, _metrics in test_ranked].index(selected_phase) + 1
        selected_test = test_metrics[selected_phase]
        best_test_phase = test_ranked[0][0]
        out.append(
            SelectionWindowResult(
                mode=mode,
                train_label=train_label,
                test_label=test_label,
                selected_phase=selected_phase,
                best_test_phase=best_test_phase,
                train_sharpe=train_metrics.sharpe,
                test_sharpe=selected_test.sharpe,
                test_return=selected_test.ret,
                test_max_dd=selected_test.max_dd,
                selected_test_rank=test_rank,
            )
        )
    return out


def regime_masks(phase_set: PhaseTargetSet) -> dict[str, np.ndarray]:
    btc_idx = phase_set.symbols.index("BTCUSDT")
    btc_ret = np.nan_to_num(phase_set.returns[:, btc_idx], nan=0.0)
    btc_vol_24h = rolling_std_prev(btc_ret, 24)
    high_vol_cutoff = np.nanquantile(btc_vol_24h, 0.75)

    finite = np.isfinite(phase_set.returns)
    counts = np.sum(finite, axis=1)
    safe_returns = np.where(finite, phase_set.returns, 0.0)
    abs_mean_market = np.divide(
        np.abs(np.sum(safe_returns, axis=1)),
        counts,
        out=np.zeros(len(counts), dtype=float),
        where=counts > 0,
    )
    mean_abs_symbol = np.divide(
        np.sum(np.abs(safe_returns), axis=1),
        counts,
        out=np.zeros(len(counts), dtype=float),
        where=counts > 0,
    )
    cohesion = abs_mean_market / (mean_abs_symbol + EPS)
    cohesion_72h = rolling_mean_prev(cohesion, 72)
    high_cohesion_cutoff = np.nanquantile(cohesion_72h, 0.75)

    gross = np.sum(np.abs(phase_set.target_weights), axis=1)
    high_gross_cutoff = np.nanquantile(gross, 0.75)

    valid_vol = np.isfinite(btc_vol_24h)
    valid_cohesion = np.isfinite(cohesion_72h)
    return {
        "high_btc_vol_top_quartile": valid_vol & (btc_vol_24h >= high_vol_cutoff),
        "low_btc_vol_bottom_3_quartiles": valid_vol & (btc_vol_24h < high_vol_cutoff),
        "high_market_cohesion_top_quartile": valid_cohesion
        & (cohesion_72h >= high_cohesion_cutoff),
        "low_market_cohesion_bottom_3_quartiles": valid_cohesion
        & (cohesion_72h < high_cohesion_cutoff),
        "high_gross_top_quartile": gross >= high_gross_cutoff,
    }


def run_regime_slices(phase_sets: dict[int, PhaseTargetSet]) -> list[RegimeResult]:
    out: list[RegimeResult] = []
    for phase, phase_set in phase_sets.items():
        for regime, mask in regime_masks(phase_set).items():
            out.append(
                RegimeResult(
                    regime=regime,
                    phase=phase,
                    bars=int(np.sum(mask)),
                    metrics=metric_snapshot(phase_set.portfolio_returns[mask]),
                )
            )
    return out


def run_execution_sensitivity(
    phase_sets: dict[int, PhaseTargetSet], *, phases: set[int]
) -> list[ExecutionSensitivityResult]:
    out: list[ExecutionSensitivityResult] = []
    for phase in sorted(phases):
        phase_set = phase_sets[phase]
        for min_notional in EXECUTION_MIN_NOTIONALS:
            result = run_execution_backtest(
                phase_set.timeline,
                phase_set.returns,
                phase_set.target_weights,
                initial_equity=INITIAL_EQUITY,
                min_order_notional=min_notional,
            )
            out.append(
                ExecutionSensitivityResult(
                    phase=phase,
                    min_order_notional=min_notional,
                    metrics=metric_snapshot(result.returns),
                    avg_turnover_per_hour=float(np.mean(result.turnover)),
                    avg_orders_per_hour=float(np.mean(result.order_counts)),
                    ignored_turnover_per_hour=float(np.mean(result.ignored_turnover)),
                    ignored_gross_avg=float(np.mean(result.ignored_gross)),
                )
            )
    return out


def neighbor_smoothness(phase_summaries: list[PhaseSummary]) -> dict[str, float]:
    by_phase = {row.phase: row for row in phase_summaries}
    p2 = by_phase[2].metrics.sharpe
    return {
        "phase_2_sharpe": p2,
        "phase_1_sharpe": by_phase[1].metrics.sharpe,
        "phase_3_sharpe": by_phase[3].metrics.sharpe,
        "phase_2_minus_neighbor_mean_sharpe": p2
        - float(np.mean([by_phase[1].metrics.sharpe, by_phase[3].metrics.sharpe])),
        "phase_2_minus_phase_0_sharpe": p2 - by_phase[0].metrics.sharpe,
        "phase_2_minus_phase_5_sharpe": p2 - by_phase[5].metrics.sharpe,
    }


def selection_summary(rows: list[SelectionWindowResult]) -> dict[str, object]:
    selected = [row.selected_phase for row in rows]
    best_test = [row.best_test_phase for row in rows]
    return {
        "windows": len(rows),
        "selected_phase_counts": {
            str(phase): selected.count(phase) for phase in sorted(set(selected))
        },
        "best_test_phase_counts": {
            str(phase): best_test.count(phase) for phase in sorted(set(best_test))
        },
        "avg_selected_test_rank": float(
            np.mean([row.selected_test_rank for row in rows])
        ),
        "avg_selected_test_sharpe": float(np.mean([row.test_sharpe for row in rows])),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Building phase target sets...")
    phase_sets = {
        phase: build_joint_target_with_core_phase(phase_hours=phase)
        for phase in range(6)
    }
    phase_summaries = [summarize_phase(phase_sets[phase]) for phase in range(6)]

    print("Running rolling/expanding phase-selection windows...")
    rolling = run_selection_windows(
        phase_sets, ROLLING_WINDOWS, mode="rolling_2y_train_1y_test"
    )
    expanding = run_selection_windows(
        phase_sets, EXPANDING_WINDOWS, mode="expanding_train_forward_test"
    )

    print("Running regime slices...")
    regimes = run_regime_slices(phase_sets)

    print("Running execution threshold sensitivity...")
    execution = run_execution_sensitivity(phase_sets, phases={0, 2, 5})

    payload = {
        "notes": {
            "scope": "Robustness audit for v16a 6h core phase candidates; no live default changes.",
            "data_requirement": "Requires full historical 1h cache, not recent-only live warmup cache.",
            "regime_method": "High-vol uses BTC 24h realized-vol top quartile; high-cohesion uses a 72h market-cohesion proxy top quartile.",
            "execution_method": "run_execution_backtest with simple min-notional thresholds, fee, and 1bp slippage.",
        },
        "phase_summaries": [
            {
                "phase": row.phase,
                "start": row.start,
                "end": row.end,
                "bars": row.bars,
                "metrics": asdict(row.metrics),
                "avg_gross": row.avg_gross,
                "avg_turnover_per_hour": row.avg_turnover_per_hour,
                "active_hours": row.active_hours,
                "order_events": row.order_events,
            }
            for row in phase_summaries
        ],
        "neighbor_smoothness": neighbor_smoothness(phase_summaries),
        "rolling_oos": [asdict(row) for row in rolling],
        "rolling_oos_summary": selection_summary(rolling),
        "expanding_oos": [asdict(row) for row in expanding],
        "expanding_oos_summary": selection_summary(expanding),
        "regime_slices": [
            {
                "regime": row.regime,
                "phase": row.phase,
                "bars": row.bars,
                "metrics": asdict(row.metrics),
            }
            for row in regimes
        ],
        "execution_threshold_sensitivity": [
            {
                "phase": row.phase,
                "min_order_notional": row.min_order_notional,
                "metrics": asdict(row.metrics),
                "avg_turnover_per_hour": row.avg_turnover_per_hour,
                "avg_orders_per_hour": row.avg_orders_per_hour,
                "ignored_turnover_per_hour": row.ignored_turnover_per_hour,
                "ignored_gross_avg": row.ignored_gross_avg,
            }
            for row in execution
        ],
    }

    out_path = OUT_DIR / "v16a_phase_robustness_research.json"
    out_path.write_text(json.dumps(payload, indent=2))

    ranked = sorted(phase_summaries, key=lambda row: row.metrics.sharpe, reverse=True)
    print("== Full-sample phase rank ==")
    for row in ranked:
        print(
            f"phase {row.phase}: sharpe={row.metrics.sharpe:.3f} "
            f"return={row.metrics.ret:.3f} maxdd={row.metrics.max_dd:.3f} "
            f"turnover/h={row.avg_turnover_per_hour:.5f}"
        )

    print("\n== Rolling OOS selection ==")
    for row in rolling:
        print(
            f"{row.train_label}->{row.test_label}: selected p{row.selected_phase}, "
            f"test sharpe={row.test_sharpe:.3f}, test rank={row.selected_test_rank}, "
            f"best test p{row.best_test_phase}"
        )

    print("\n== Expanding OOS selection ==")
    for row in expanding:
        print(
            f"{row.train_label}->{row.test_label}: selected p{row.selected_phase}, "
            f"test sharpe={row.test_sharpe:.3f}, test rank={row.selected_test_rank}, "
            f"best test p{row.best_test_phase}"
        )

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
