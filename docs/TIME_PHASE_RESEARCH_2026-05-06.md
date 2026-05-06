# Time-phase research note — 2026-05-06

Objective: investigate whether the current UTC-aligned execution / sampling times
for v16a leave avoidable edge on the table, while staying disciplined about
explainability and overfitting risk.

## Scope and boundary

This study covers only UTC-hour phase choices that the current data can actually
resolve.

What is in scope now:
- 6h core sleeve phase shifts (0..5 UTC-hour offsets), rebuilt from 1h bars.
- Descriptive hour-of-day contribution audit for the current 1h overlay.
- Simple hour-prior counterfactuals used as sanity checks, not production
  selectors.

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
- `backtest-results/v16a_time_phase_research.png`

Method:
1. Rebuild synthetic phased 6h bars from 1h parquet data.
2. Recompute the shifted v10g core sleeve for each phase `0..5`.
3. Keep the current 1h overlay logic unchanged.
4. Rebuild the joint v16a target stream and compare full-sample + split metrics.
5. Separately audit overlay contribution by UTC hour as a descriptive signal.

## Current findings after full-history cache rerun

### 6h core phase sweep

Full-sample ranking by Sharpe after rerunning against the full historical 1h
cache:
- phase `2`: Sharpe `2.287`, return `1.571`, max drawdown `0.055`.
- phase `0`: Sharpe `2.200`, return `1.515`, max drawdown `0.065`.
- phase `5`: Sharpe `2.158`, return `1.491`, max drawdown `0.065`.
- phase `1`: Sharpe `2.127`, return `1.400`, max drawdown `0.054`.

Split behavior:
- 2020-2021: phase `2` strongest, with phase `5` close behind.
- 2022-2023: phase `0` strongest, phase `2` second.
- 2024-2026: phase `3` strongest, but phase `0` and `2` are close and phase
  `2` has the cleanest drawdown among those three.

Mean rank across full sample plus the three calendar splits:
- phase `2`: `1.75`
- phase `0`: `2.25`
- phase `5`: `3.25`
- phase `1`: `3.75`

Small walk-forward phase selection also picked phase `2` in both tests:
- train 2020-2021 -> test 2022-2023: selected phase `2`, test Sharpe `1.260`.
- train 2020-2023 -> test 2024-2026: selected phase `2`, test Sharpe `1.846`.

Interpretation:
- There is real phase sensitivity.
- The updated full-history evidence now favors phase `2` more clearly than the
  first pass did.
- The margin over phase `0` is still not large enough to justify a live change by
  itself. Treat phase `2` as the leading research candidate, not a production
  default yet.
- Phase `0` remains a strong baseline and is still the best split performer in
  2022-2023.

### Overlay hour prior audit

The current overlay already has a UTC-hour prior (`favorable_hours` /
`avoid_hours`). The descriptive audit should be treated as a falsification tool,
not as a production selector by itself.

Current descriptive audit highlights:
- Strong positive hours not currently favored: `09`, `21`, `01`, `18`, `03`.
- Supported current favorable hour: `06`.
- Weak but positive current favorable hours: `13`, `17`.
- Directionally contradicted current favorable hours: `14`, `20`.
- Current avoid hours `15` and `16` were not obviously bad in this descriptive
  view; `16` was modestly positive.

Simple counterfactual hour-prior sanity check for phase `2`:
- current prior: joint Sharpe `2.287`, return `1.571`, max drawdown `0.055`.
- neutral `0.75` multiplier for all hours: joint Sharpe `2.191`, return `1.481`,
  max drawdown `0.054`.
- neutral `1.00` multiplier for all hours: joint Sharpe `1.635`, return `1.056`,
  max drawdown `0.061`.

Interpretation:
- The current differentiated prior still beats a neutral all-hour prior in the
  full-history phase-`2` joint test.
- The descriptive hour table still falsifies parts of the hand-picked prior,
  especially `14` and `20` as favorable hours.
- Because hour attribution is path-dependent and the prior changes trade
  selection, do not replace the current prior with a direct top-hour selector.

## Recommended next steps

1. Keep live unchanged for now.
2. Extend the study with stricter robustness checks:
   - rolling OOS windows,
   - neighboring-phase smoothness checks,
   - turnover / order-count / notional-threshold sensitivity,
   - per-phase performance under high-vol / high-correlation regimes.
3. If phase `2` keeps showing a materially better stress trade-off, implement a
   research-only parameterized `core_phase_hours` path before any production
   move.
4. Re-test the hour prior with rolling OOS and symbol-level attribution before
   changing `favorable_hours` / `avoid_hours`.
5. Do not optimize minute-level offsets until sub-hour data exists.
