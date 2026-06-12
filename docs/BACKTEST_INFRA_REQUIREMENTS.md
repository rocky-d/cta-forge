# CTA-Forge Backtest Infra — Consolidated Requirements

Last update: 2026-06-12

---

## 1. Motivation

21 research scripts (`scripts/backtest/`, 8320 lines total) each carry ~60%+ identical boilerplate:

- Data loading & caching
- Signal pipeline orchestration
- Target-set construction
- Simulation execution
- Metric computation
- Chart generation
- Artifact output

This document defines the requirements for a `libs/backtest` shared library that absorbs the boilerplate, keeping research scripts thin orchestrators (~100 lines) and the infra reusable.

---

## 2. Non-Goals (What `libs/backtest` is NOT)

- NOT a live trading engine or execution path — that stays in executor
- NOT a strategy authoring framework — strategies live in executor profiles
- NOT a data pipeline — data loading/fetching stays in data-service
- NOT a web API or service — it's a pure library with no runtime
- NOT a replacement for report-service plot.py — charting here is for research PNGs, not dashboard

---

## 3. Core Architecture (as built)

```
libs/backtest/
├── engine.py        ← Target-weight & execution-aware simulators
├── metrics.py       ← Unified metric computation
├── result.py        ← Dataclass types (TargetBacktestResult, ExecutionBacktestResult, BacktestMetrics, ChartSeries)
├── chart.py         ← 3-panel comparison chart generation
├── experiment.py    ← Experiment definition & runner (+ DataConfig, TimeRangeConfig, AccountConfig, StrategyConfig, OutputConfig)
└── __init__.py
```

Dependency chain: `backtest` → `core` + `numpy` + `polars` + `matplotlib`. No dependency on executor/alpha-service/data-service profiles or strategies.

`data.py` was spec'd but intentionally omitted — see §4.1 decision note.

---

## 4. Requirements by Domain

### 4.1 Data Source Configuration

**Spec**: Multi-path `DataConfig` with staleness thresholds, `data.py` helper module.

**Decision (2026-06-12)**: Not built. The actual codebase uses a single `DATA_DIR → ParquetStore` pattern. Multi-path search and staleness thresholds are solutions looking for problems — no script today has a multi-path data loading scenario. If data is stale the backtest produces visibly wrong results (loud failure > silent check). Constitution: _less is more_, _let errors surface before adding fallbacks_, _avoid premature abstraction_.

**Built**: `DataConfig` exists in `experiment.py` with `path`, `timeframe`, `min_bars_per_symbol` — sufficient for current usage.

### 4.2 Symbol Universe

**Spec**: `SymbolConfig` with named universe registry, exclude lists, perp-since filters.

**Decision (2026-06-12)**: Not built. Symbol lists are already defined as constants in `core/constants.py` and `v16a_badscore_overlay.py`. Adding a registry layer turns a 1-line import into a config+registry maintenance burden. If universe naming conventions are ever needed they can be added as simple dicts in constants — no new module required. Constitution: _less is more_, _avoid premature abstraction_.

### 4.3 Time Range

**Spec**: `TimeRangeConfig` with OOS demarcation (`oos_start`, `oos_label`).

**Decision (2026-06-12)**: OOS support not built. OOS splitting varies too much between studies (rolling windows, expanding windows, fixed splits) to benefit from a generic config field. Forcing scripts through a unified OOS API would increase coupling not reduce it. Constitution: _high cohesion, low coupling_.

**Built**: `TimeRangeConfig` with `start`, `end`, `warmup_bars` — the stable subset that every script needs.

### 4.4 Initial Capital & Account

**Built as spec'd**: `AccountConfig` with `initial_equity`, `fee`, `slippage`, `min_order_notional`.

### 4.5 Strategy Configuration

**Built**: `StrategyConfig` with explicit fields (`gross_cap`, `target_scale`, `dd_circuit_breaker`, `core_phase_hours`, `gate_rolling_years`, `v10g_allocation`, `overlay_allocation`).

**Decision (2026-06-12)**: The spec's `v10g_params: dict[str, Any]` override was intentionally omitted — it creates a fuzzy interface where callers pass arbitrary dicts. Explicit fields are clearer and self-documenting. Constitution: _less is more_, _high cohesion, low coupling_.

### 4.6 Simulation Engine Selection

Implemented as two separate engine functions (`run_target_weight_backtest` vs `run_execution_backtest`). Callers pick by calling the function they need. No runtime engine selection string — the function IS the selection.

### 4.7 Experiment Definition

**Built**: `BacktestExperiment` in `experiment.py` with `run_backtest()` and `save_experiment_artifacts()`. No `compute_metrics()` method on the experiment itself — metrics computation is a separate call on the result.

### 4.8 Multi-Experiment / Parameter Sweep

**Decision (from original Phase 1 scoping)**: Deferred. The individual experiment runner eliminates ~80% of boilerplate. `ExperimentSweep` will be reconsidered only if 5+ scripts are essentially manual sweeps — at that point the abstraction pays for itself.

### 4.9 Metrics — Unified Computation

**Built as spec'd**: `compute_metrics()` in `metrics.py` covering total/ann return, vol, Sharpe/Sortino/Calmar, maxDD/maxDD duration/avgDD/ulcer, tail ratio (P95/P5), portfolio-level gross/net exposure + turnover. Single `BacktestMetrics` dataclass output.

Legacy `calculate_hourly_metrics()` kept as backward-compat wrapper.

### 4.10 Charts — Decoupled Rendering

**Built as spec'd**: 16:9 3-panel (equity/drawdown/monthly bar), max 4 configs, `ChartSeries` data contract, `PanelSpec` for panel customization.

### 4.11 Output & Artifact Management

**Built**: `OutputConfig` with `base_dir`, `subdir` (auto-generated if None). `save_experiment_artifacts()` writes `metrics.json` + `experiment.json`.

**Decision (2026-06-12)**: The spec's `save_equity_csv`, `save_metrics_json`, `save_chart_png`, `save_config_json` boolean toggles were omitted. No caller has needed to suppress any artifact type. Constitution: _less is more_, _avoid premature abstraction_.

### 4.12 Caching & Incremental Computation

**Decision (Phase 2)**: Not built. Rustrated. 2-5 minute backtest runtime is acceptable for current usage patterns. Will revisit if parameter sweeps with 20+ combos become routine.

### 4.13 Progress Reporting

**Decision (2026-06-12)**: To be implemented simply. No tqdm dependency — `print()` statements at pipeline stage boundaries. This is the one Phase 1 gap that addresses a real UX pain point (2-5 minute silent runs) at near-zero cost. Constitution: _pragmatism and iteration_.

### 4.14 Reproducibility

**Decision (2026-06-12)**: Git commit hash will be captured in `experiment.json`. This is ~3 lines of code and adds traceability to the exact code version. Python version / package list capture is over-engineering — not needed.

### 4.15 Integration & Migration Path

Status: Phase 1 library coexists with existing code. `portfolio_backtest.py` is a re-export shim. No scripts migrated yet (Phase 2).

---

## 5. Variable Catalog

Covered in detail in `.tmp-backtest-vars.md`. ~70 configurable variables across 7 categories.

Key takeaway for infra design: the experiment config dataclasses above must be able to express any combination of these 70 variables without the user needing to understand all 70. Sensible defaults for everything; only override what changes.

---

## 6. CONSTITUTION Review

### ✅ Aligned

| Principle | How |
|---|---|
| **Less is more** | 5 modules (not 6 as spec'd — data.py intentionally omitted). No new dependencies. |
| **High cohesion, low coupling** | Each module has one clear concern; chart knows nothing about strategies; metrics knows nothing about engines. |
| **Pragmatism and iteration** | Phase 1 shipped the core. Gaps evaluated against constitution before filling. |
| **Research before action** | This document + post-implementation review. What was spec'd vs what was actually needed. |
| **First principles** | Questioned spec items individually — not "the spec says so, therefore we must". |
| **Avoid premature abstraction** | SymbolConfig, multi-path DataConfig, OOS in TimeRangeConfig, OutputConfig toggles — all rejected on this principle. |
| **Let errors surface before adding fallbacks** | Staleness checks omitted. Bad data produces loud failures. |
| **Programming design** | Composition over inheritance — ChartSeries, PanelSpec composed into figures; experiment configs composed into runs. |

### ⚠️ Potential Concerns (revisited)

| Concern | Status |
|---|---|
| **Scope creep** | Mitigated. 4 of 5 Phase 1 gaps intentionally rejected. |
| **Over-engineering** | Mitigated. ExperimentSweep deferred. Spec simplifications applied. |
| **Backward compat** | Stable. portfolio_backtest.py shim unchanged. |

---

## 7. Implementation Phasing

### Phase 1 — Core Library ✅ DONE (2026-05 ~ 2026-06)

- [x] `libs/backtest/` workspace member with `pyproject.toml`
- [x] `engine.py` — `run_target_weight_backtest` + `run_execution_backtest`
- [x] `result.py` — `TargetBacktestResult`, `ExecutionBacktestResult`, `BacktestMetrics`, `ChartSeries`
- [x] `metrics.py` — unified `compute_metrics()` (Tier 1 + portfolio-level)
- [x] `chart.py` — `create_comparison_figure()`, `save_figure()`, 3-panel default
- [x] `experiment.py` — `BacktestExperiment`, config dataclasses, `run_backtest()`, `save_experiment_artifacts()`
- [x] `portfolio_backtest.py` re-export shim for backward compat
- [ ] ~~`data.py` unified loading helper~~ — intentionally omitted (§4.1)
- [ ] ~~`SymbolConfig`~~ — intentionally omitted (§4.2)
- [ ] ~~OOS support in `TimeRangeConfig`~~ — intentionally omitted (§4.3)

**Phase 1.1 — Finish-up** (2026-06-12)

- [ ] Progress reporting: `print()` stage-boundary logs in pipeline
- [ ] Git commit hash in `experiment.json`

### Phase 2 — Integration & Polish

- Migrate 1-2 research scripts as pilot
- Cache layer for target-set computation
- `ExperimentSweep` if justified by actual usage (5+ manual-sweep scripts)
- Additional Tier 2 metrics (Sterling, skewness, kurtosis)

### Phase 3 — Full Migration

- Migrate remaining active scripts
- Archive unused scripts
- Remove backward compat shims from executor
- Tier 3+ metrics as needed

---

## 8. Open Questions (for Boss)

1. ~~**OK to drop `ExperimentSweep`** from Phase 1?~~ ✅ Confirmed: deferred.

2. ~~**Cache layer priority**~~ ✅ Phase 2. 2-5 min per run acceptable for now.

3. ~~**Symbol universe naming**~~ ✅ Moot — SymbolConfig entirely omitted. Use existing constants in `core/constants.py`.

4. ~~**Backward compat timeline**~~ ✅ Keep shim until all scripts migrated (Phase 3).
