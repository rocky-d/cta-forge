# Time-phase research note — 2026-05-06

Objective: investigate whether the current UTC-aligned execution / sampling times
for v16a leave avoidable edge on the table, while staying disciplined about
explainability and overfitting risk.

## Scope and boundary

This first pass studies only UTC-hour phase choices that the current data can
actually resolve.

What is in scope now:
- 6h core sleeve phase shifts (0..5 UTC-hour offsets), rebuilt from 1h bars.
- Descriptive hour-of-day contribution audit for the current 1h overlay.

What is not in scope yet:
- Minute-level offsets such as `HH:17` or `HH:43`.
- Those would require sub-hour data. With the current 1h/6h bars, pretending to
  optimize minute offsets would be fake precision.

## Why this is not obviously overfitting

A time-phase hypothesis can be structurally explainable:
- US cash open / close and macro release windows can cluster volatility.
- Cross-session liquidity transitions (Asia → Europe → US) can change signal
  quality and slippage conditions.
- A 6h phase shift changes which information is packed into each core trend bar;
  that is a real modelling choice, not just a cosmetic timestamp tweak.

The overfitting risk is still high, so we prefer:
- small candidate sets,
- stability across splits,
- neighboring-phase smoothness,
- explainability over single-point in-sample maximization.

## Current code-level reality

Relevant current bindings:
- `LiveEngine._wait_for_candle_close()` waits for the next exact timeframe close
  on UTC hour boundaries.
- `v16a_badscore_overlay.build_v10g_sleeve()` currently uses canonical `6h`
  parquet bars.
- `build_v16a_target_set()` combines:
  - shifted 6h v10g core,
  - 1h overlay,
  - hour-by-hour forward fill of the latest 6h core sleeve.
- The overlay already contains a deterministic UTC-hour soft prior inside
  `local_overlay_signal()`.

Implication:
- A live 6h phase change is feasible, but it is not a one-line env tweak.
- The cleanest implementation path would be to synthesize phased 6h bars from
  the 1h cache inside the target builder (or persist phased derived caches), and
  thread a `core_phase_hours` parameter through:
  - target construction,
  - preflight,
  - live wait logic,
  - diagnostics/report metadata.
- Minute-level offset research should wait until sub-hour data exists.

## Research harness

Repo-native reproduction script:

```bash
uv run python scripts/backtest/v16a_time_phase_research.py
```

It writes:
- `backtest-results/v16a_time_phase_research.json`

Method:
1. Rebuild synthetic phased 6h bars from 1h parquet data.
2. Recompute the shifted v10g core sleeve for each phase `0..5`.
3. Keep the current 1h overlay logic unchanged.
4. Rebuild the joint v16a target stream and compare full-sample + split metrics.
5. Separately audit overlay contribution by UTC hour as a descriptive signal.

## First-pass findings

### 6h core phase sweep

Full-sample ranking by Sharpe from the first pass:
- phase `0` best overall return / Sharpe.
- phase `1` slightly lower return, but lower drawdown.
- phase `2` lower return again, but the cleanest drawdown among the top three.
- phases `3/4/5` materially weaker, especially `4`.

Split behavior from the first pass:
- 2020-2021: phase `0` strongest.
- 2022-2023: phase `1/2` more resilient.
- 2024-2026: phase `0` strongest, phase `2` second.

Interpretation:
- There is real phase sensitivity.
- The evidence does **not** currently say “just move away from UTC-aligned bars
  and everything gets better”.
- The interesting trade-off is not raw return dominance, but whether phase `1/2`
  offer a better stress-regime compromise than phase `0`.

### Overlay hour prior audit

The current overlay already has a UTC-hour prior (`favorable_hours` /
`avoid_hours`). The descriptive audit should be treated as a falsification tool,
not as a production selector by itself.

If an hour prior looks directionally inconsistent with realized contribution,
that is a prompt for deeper research — not an immediate live change.

## Recommended next steps

1. Keep live unchanged for now.
2. Extend the study with stricter robustness checks:
   - rolling OOS windows,
   - neighboring-phase smoothness checks,
   - turnover / order-count / notional-threshold sensitivity,
   - per-phase performance under high-vol / high-correlation regimes.
3. If phase `1` or `2` keep showing a materially better stress trade-off,
   implement a research-only parameterized `core_phase_hours` path before any
   production move.
4. Do not optimize minute-level offsets until sub-hour data exists.
