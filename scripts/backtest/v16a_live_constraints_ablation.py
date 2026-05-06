"""Ablate live-like v16a pilot execution constraints.

This research script answers why current pilot-style execution differs from the
ideal target-weight backtest. It keeps strategy targets fixed and varies account
equity, minimum order notional, max increase order notional, target scale, and
gross cap.

Run:
    uv run python scripts/backtest/v16a_live_constraints_ablation.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from executor.portfolio_backtest import (  # noqa: E402
    calculate_hourly_metrics,
    run_target_weight_backtest,
)
from executor.profiles.v16a_badscore_overlay import normalize_gross  # noqa: E402
from scripts.backtest.v16a_phase_robustness_research import (  # noqa: E402
    build_joint_target_with_core_phase,
)

OUT = ROOT / "backtest-results" / "v16a_live_constraints_ablation.json"
PHASES = [0, 2]
TARGET_SCALE = 5.0
TARGET_GROSS_CAP = 4.0
FEE = 0.0004
EPS = 1e-12


@dataclass(frozen=True)
class Case:
    name: str
    initial_equity: float = 100.0
    min_order: float = 0.0
    max_increase: float | None = None
    fee: float = FEE
    slippage: float = 0.0
    target_scale: float = TARGET_SCALE
    gross_cap: float = TARGET_GROSS_CAP
    mode: str = "exec"  # exec or target


@dataclass(frozen=True)
class Metrics:
    case: str
    phase: int
    initial_equity: float
    min_order: float
    max_increase: float | None
    fee: float
    slippage: float
    target_scale: float
    gross_cap: float
    ret: float
    ann_return: float
    volatility: float
    sharpe: float
    max_dd: float
    final_equity: float
    mean_realized_gross: float
    p95_realized_gross: float
    max_realized_gross: float
    avg_orders_per_hour: float
    avg_turnover_per_hour: float
    avg_ignored_gross: float


def scale_weights(
    base: np.ndarray, symbols: list[str], *, scale: float, cap: float
) -> np.ndarray:
    out = np.zeros_like(base, dtype=float)
    for i in range(base.shape[0]):
        scaled = {sym: float(base[i, j]) * scale for j, sym in enumerate(symbols)}
        capped = normalize_gross(scaled, gross_cap=cap)
        out[i] = np.array([capped.get(sym, 0.0) for sym in symbols])
    return out


def apply_caps(current, target, equity, *, min_order, max_increase):
    nxt = current.copy()
    turnover = 0.0
    orders = 0
    ignored_turnover = 0.0

    def leg(idx, delta, *, reduce_only):
        nonlocal turnover, orders, ignored_turnover
        if abs(delta) <= EPS:
            return
        notional = delta * equity
        if not reduce_only and max_increase is not None and max_increase > 0:
            notional = max(-max_increase, min(max_increase, notional))
            delta = notional / equity
        if abs(notional) < min_order:
            ignored_turnover += abs(delta)
            return
        nxt[idx] += delta
        turnover += abs(delta)
        orders += 1

    for idx, desired in enumerate(target):
        cur = float(nxt[idx])
        tgt = float(desired)
        if abs(cur) > EPS and cur * tgt < 0:
            leg(idx, -cur, reduce_only=True)
            leg(idx, tgt, reduce_only=False)
        else:
            delta = tgt - cur
            reduce_only = abs(cur) > EPS and abs(tgt) < abs(cur)
            leg(idx, delta, reduce_only=reduce_only)
    ignored_gross = float(np.sum(np.abs(target - nxt)))
    return nxt, turnover, orders, ignored_turnover, ignored_gross


def run_exec(timeline, returns, target, case: Case):
    pnl = np.zeros(target.shape[0])
    turnover = np.zeros(target.shape[0])
    orders = np.zeros(target.shape[0], dtype=int)
    ignored = np.zeros(target.shape[0])
    realized = np.zeros_like(target)
    cur = np.zeros(target.shape[1])
    equity = case.initial_equity
    for t in range(target.shape[0] - 1):
        cur, turnover[t], orders[t], _ignored_turnover, ignored[t] = apply_caps(
            cur,
            np.nan_to_num(target[t], nan=0.0),
            equity,
            min_order=case.min_order,
            max_increase=case.max_increase,
        )
        realized[t] = cur
        market = float(np.sum(cur * np.nan_to_num(returns[t + 1], nan=0.0)))
        pnl[t + 1] = market - turnover[t] * (case.fee + case.slippage)
        equity *= 1.0 + pnl[t + 1]
    realized[-1] = cur
    equity_curve = case.initial_equity * np.cumprod(1.0 + pnl)
    return pnl, equity_curve, turnover, orders, ignored, realized


def calc(case: Case, phase_data):
    out = []
    for phase, ps in phase_data.items():
        target = scale_weights(
            ps.target_weights, ps.symbols, scale=case.target_scale, cap=case.gross_cap
        )
        if case.mode == "target":
            bt = run_target_weight_backtest(
                ps.timeline,
                ps.returns,
                target,
                initial_equity=case.initial_equity,
                fee=case.fee + case.slippage,
            )
            pnl = bt.returns
            equity = case.initial_equity * np.cumprod(1.0 + pnl)
            turnover = bt.turnover
            orders = np.zeros(len(pnl), dtype=int)
            ignored = np.zeros(len(pnl))
            realized = target
        else:
            pnl, equity, turnover, orders, ignored, realized = run_exec(
                ps.timeline, ps.returns, target, case
            )
        m = calculate_hourly_metrics(pnl, initial_equity=case.initial_equity)
        gross = np.sum(np.abs(realized), axis=1)
        out.append(
            Metrics(
                case=case.name,
                phase=phase,
                initial_equity=case.initial_equity,
                min_order=case.min_order,
                max_increase=case.max_increase,
                fee=case.fee,
                slippage=case.slippage,
                target_scale=case.target_scale,
                gross_cap=case.gross_cap,
                ret=float(m["return"]),
                ann_return=float(m["ann_return"]),
                volatility=float(m["volatility"]),
                sharpe=float(m["sharpe"]),
                max_dd=float(m["max_dd"]),
                final_equity=float(equity[-1]),
                mean_realized_gross=float(np.mean(gross)),
                p95_realized_gross=float(np.quantile(gross, 0.95)),
                max_realized_gross=float(np.max(gross)),
                avg_orders_per_hour=float(np.mean(orders)),
                avg_turnover_per_hour=float(np.mean(turnover)),
                avg_ignored_gross=float(np.mean(ignored)),
            )
        )
    return out


def main():
    phase_data = {p: build_joint_target_with_core_phase(phase_hours=p) for p in PHASES}
    cases = [
        Case(
            "theoretical_target_weight_fee_only",
            initial_equity=100,
            mode="target",
            min_order=0,
            max_increase=None,
            fee=0.0004,
            slippage=0.0,
        ),
        Case(
            "target_weight_fee_plus_slippage",
            initial_equity=100,
            mode="target",
            min_order=0,
            max_increase=None,
            fee=0.0004,
            slippage=0.0001,
        ),
        Case(
            "execution_no_min_no_max_fee_only",
            initial_equity=100,
            min_order=0,
            max_increase=None,
            fee=0.0004,
            slippage=0.0,
        ),
        Case(
            "execution_no_min_no_max_fee_plus_slippage",
            initial_equity=100,
            min_order=0,
            max_increase=None,
            fee=0.0004,
            slippage=0.0001,
        ),
        Case(
            "min10_only",
            initial_equity=100,
            min_order=10,
            max_increase=None,
            fee=0.0004,
            slippage=0.0001,
        ),
        Case(
            "max50_only",
            initial_equity=100,
            min_order=0,
            max_increase=50,
            fee=0.0004,
            slippage=0.0001,
        ),
        Case(
            "live_like_min10_max50",
            initial_equity=100,
            min_order=10,
            max_increase=50,
            fee=0.0004,
            slippage=0.0001,
        ),
    ]
    # Sensitivity cases.
    for eq in [100, 150, 200, 300, 500, 1000]:
        cases.append(
            Case(
                f"equity{eq:g}_min10_max50",
                initial_equity=eq,
                min_order=10,
                max_increase=50,
                fee=0.0004,
                slippage=0.0001,
            )
        )
    for max_inc in [25, 50, 75, 100, 150, 200, None]:
        label = "none" if max_inc is None else f"{max_inc:g}"
        cases.append(
            Case(
                f"equity100_min10_max{label}",
                initial_equity=100,
                min_order=10,
                max_increase=max_inc,
                fee=0.0004,
                slippage=0.0001,
            )
        )
    for scale in [3, 4, 5, 6, 7]:
        cases.append(
            Case(
                f"equity100_scale{scale:g}_gross4",
                initial_equity=100,
                min_order=10,
                max_increase=50,
                fee=0.0004,
                slippage=0.0001,
                target_scale=scale,
                gross_cap=4.0,
            )
        )
    for cap in [1.5, 2, 3, 4]:
        cases.append(
            Case(
                f"equity100_scale5_gross{cap:g}",
                initial_equity=100,
                min_order=10,
                max_increase=50,
                fee=0.0004,
                slippage=0.0001,
                target_scale=5.0,
                gross_cap=cap,
            )
        )

    metrics = []
    for c in cases:
        metrics.extend(calc(c, phase_data))
    payload = {"cases": [asdict(m) for m in metrics]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")
    # Print compact selected table.
    selected = {
        "theoretical_target_weight_fee_only",
        "target_weight_fee_plus_slippage",
        "execution_no_min_no_max_fee_plus_slippage",
        "min10_only",
        "max50_only",
        "live_like_min10_max50",
    }
    for m in metrics:
        if m.case in selected:
            print(
                f"{m.case:42s} p{m.phase} ret={m.ret:8.3f} ann={m.ann_return:6.3f} sharpe={m.sharpe:5.3f} dd={m.max_dd:5.3f} orders={m.avg_orders_per_hour:6.3f} ignored={m.avg_ignored_gross:6.3f} gross={m.mean_realized_gross:6.3f}"
            )


if __name__ == "__main__":
    main()
