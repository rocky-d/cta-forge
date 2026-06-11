# CTA-Forge Backtest Infra — Consolidated Requirements

Last update: 2026-06-11

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

## 3. Core Architecture

```
libs/backtest/
├── engine.py        ← Target-weight & execution-aware simulators
├── metrics.py       ← Unified metric computation
├── result.py        ← Dataclass types (BacktestMetrics, BacktestResult, ChartSeries)
├── data.py          ← Data provisioning helpers (not pipeline, just unified loading)
├── chart.py         ← 3-panel comparison chart generation
├── experiment.py    ← Experiment definition & runner
└── __init__.py
```

Dependency chain: `backtest` → `core` + `numpy` + `polars` + `matplotlib`. No dependency on executor/alphaservice/data-service profiles or strategies.

---

## 4. Requirements by Domain

### 4.1 Data Source Configuration

**Problem**: Scripts scatter data paths everywhere — `data/`, `/tmp/cta-forge-data-backup`, `/tmp/cta-forge-data-extended`, `data-minute-offset-15m/`. No standard way to specify which data to use.

**Requirements**:
- Declarative data spec: path + timeframe + format (parquet)
- Support multiple data directories (e.g. `data/` for perp-only, a separate dir for spot data)
- Data freshness check: verify bar count, latest timestamp, symbol coverage before running
- Caching layer: avoid re-downloading if cache is fresh enough

```python
@dataclass
class DataConfig:
    paths: list[str | Path]            # ordered search path; first match wins
    timeframe: str = "6h"              # "1h", "6h", "15m"
    min_bars_per_symbol: int = 500     # reject symbols with fewer bars
    stale_threshold_hours: int = 24    # warn if latest bar is older than this
```

### 4.2 Symbol Universe

**Problem**: Each script hardcodes its own symbol list. No standard way to define, filter, or validate universes. 19-default, top-10, majors-5, extended-25 — all ad-hoc.

**Requirements**:
- Named universe registry: `{"baseline_19": [...], "top10": [...], "majors_5": [...]}`
- Declarative universe as part of experiment config
- Validation: reject symbols that don't exist in data source
- Support custom inline lists for one-off research

```python
@dataclass
class SymbolConfig:
    universe: str | list[str]          # named universe or explicit list
    exclude: list[str] | None = None   # symbols to exclude
    require_perp_since: datetime | None = None  # filter: must have data since X
```

### 4.3 Time Range

**Problem**: Scripts hardcode start dates (2019-09-01), compute end dates from data, and have inconsistent warmup handling.

**Requirements**:
- Configurable start/end with sensible defaults
- Warmup bar count (default 150, but tunable)
- Support phase demarcation (before/after live start date)
- Support out-of-sample split specification

```python
@dataclass
class TimeRangeConfig:
    start: datetime | None = None      # None = earliest available
    end: datetime | None = None        # None = latest available
    warmup_bars: int = 150
    oos_start: datetime | None = None  # out-of-sample start (phase boundary)
    oos_label: str = "OOS"             # label for OOS period in charts
```

### 4.4 Initial Capital & Account

**Problem**: `INITIAL_EQUITY = 10_000.0` hardcoded in multiple places. Some scripts scale it, others don't.

**Requirements**:
- Per-experiment initial capital setting
- Optional equity scale factor (multiply all weights/returns by this)
- Fee/slippage config as part of simulation params

```python
@dataclass
class AccountConfig:
    initial_equity: float = 10_000.0
    fee: float = 0.0004
    slippage: float = 0.0001
    min_order_notional: float = 10.0
```

### 4.5 Strategy Configuration

**Problem**: Strategy params are spread across V10GStrategyParams (24 fields), v16a profile params, and profile-specific helper functions. No standard way to specify "run with these strategy params".

**Requirements**:
- V10GStrategyParams as a configurable object (already a dataclass, good)
- v16a profile params (allocation, cap, phase, gate) as a separate config block
- Profile selection: `v10g-engine` vs `v16a-badscore-overlay`
- Parameter override mechanism: start from defaults, override specific fields

```python
@dataclass
class StrategyConfig:
    profile: str = "v16a-badscore-overlay"
    v10g_params: dict[str, Any] | None = None  # override specific V10G fields
    v16a_allocation: float = 0.5
    v16a_gross_cap: float = 1.0
    v16a_core_phase_hours: int = 0
    v16a_gate_rolling_years: float = 0.0
    target_scale: float = 1.0
    dd_circuit_breaker: float = 0.08
    leverage_multiplier: int = 1
```

### 4.6 Simulation Engine Selection

**Problem**: We have two simulators but scripts sometimes use one, sometimes the other, sometimes both. No standard way to pick.

**Requirements**:
- Explicit engine selection: `target_weight` (vectorized, fast) or `execution_aware` (order simulation)
- Target-weight is the default for portfolio-level research
- Execution-aware only when min-order-notional / sign-flip effects matter

### 4.7 Experiment Definition — The Unifying Config

All the above configs roll up into one experiment definition:

```python
@dataclass
class BacktestExperiment:
    name: str                          # human-readable label
    data: DataConfig
    symbols: SymbolConfig
    time_range: TimeRangeConfig
    account: AccountConfig
    strategy: StrategyConfig
    engine: str = "target_weight"      # "target_weight" | "execution_aware"
    output: OutputConfig | None = None

    def run(self) -> BacktestResult: ...
    def compute_metrics(self, result: BacktestResult) -> BacktestMetrics: ...
```

This is inspired by the CONSTITUTION principle "high cohesion, low coupling" — the experiment definition is a pure data object that can be serialized, compared, and version-controlled. The `run()` method delegates to the engine; the engine doesn't know about the experiment format.

### 4.8 Multi-Experiment / Parameter Sweep

Group of experiments sharing most config, varying one axis:

```python
@dataclass
class ExperimentSweep:
    base: BacktestExperiment
    vary: str                          # field name to vary, e.g. "strategy.dd_circuit_breaker"
    values: list[Any]                  # [0.0, 0.08, 0.12, 0.16]
    labels: list[str] | None = None    # optional human labels per value

    def run_all(self) -> list[BacktestResult]: ...
    def compare(self, results: list[BacktestResult]) -> Figure: ...
```

This directly replaces the pattern in gate_dd_ablation.py where the same function is called in a loop over DD values. The sweep runner handles parallelism, progress reporting, and result collection.

### 4.9 Metrics — Unified Computation

Covered in detail in `.tmp-backtest-metrics.md`. Summary:

- One function: `compute_metrics(returns, weights=None, trades=None) → BacktestMetrics`
- Phase 1: all Tier 1 core metrics + portfolio essentials (gross/net exposure, turnover, tail ratio, max dd duration)
- Single dataclass output, not scattered dicts
- Supports both weight-driven and trade-driven inputs

### 4.10 Charts — Decoupled Rendering

Covered in detail in `.tmp-backtest-charts.md`. Summary:

- Default: 16:9 vertical 3-panel (equity / drawdown / monthly bar)
- Max 4 configs per chart
- ChartSeries data contract decouples rendering from computation
- PanelSpec allows adding custom panels without modifying chart module
- Saved to PNG via `save_figure(fig, path)`

### 4.11 Output & Artifact Management

**Problem**: Each script manually creates directories in `backtest-results/` with inconsistent naming. Some use timestamps, some don't. No standard artifact naming.

**Requirements**:
- Experiment-run auto-creates output directory: `backtest-results/{experiment_name}_{timestamp}/`
- Standard artifact names: `metrics.json`, `equity.csv`, `chart.png`, `experiment.json` (config snapshot)
- Config snapshot for reproducibility: save the full experiment config as JSON alongside results
- Optional: git commit hash in experiment.json for traceability

```python
@dataclass
class OutputConfig:
    base_dir: str | Path = "backtest-results"
    subdir: str | None = None           # None = auto-generate from experiment name + timestamp
    save_equity_csv: bool = True
    save_metrics_json: bool = True
    save_chart_png: bool = True
    save_config_json: bool = True        # reproducibility
    chart_filename: str = "chart.png"
    metrics_filename: str = "metrics.json"
    equity_filename: str = "equity.csv"
```

### 4.12 Caching & Incremental Computation

**Problem**: The full pipeline (load data → precompute → signals → target set → simulate) takes minutes. When only one parameter changes (e.g. gross_cap), the first three stages are identical. No caching.

**Requirements**:
- Stage-based caching: each pipeline stage can be cached independently
- Cache key based on input hash (data config + symbol config + strategy params that affect that stage)
- Target-set is the heaviest stage to recompute; it should be cacheable
- Simulation (varying cap/scale/leverage) is cheap and doesn't need caching
- Cache location: `backtest-results/.cache/` or project-level `.backtest-cache/`

Priority: Phase 2. Not needed for initial release. The CONSTITUTION says "let errors surface before adding fallbacks" — simple caching is fine, but complex dependency tracking is over-engineering at this stage.

### 4.13 Progress Reporting

Backtests can take 2-5 minutes. Silence is bad UX.

**Requirements**:
- `tqdm`-style progress bar at pipeline level (optional, controlled by verbose flag)
- Stage-level logging: "Loading data...", "Precomputing indicators...", "Building target set...", "Running simulation...", "Computing metrics...", "Generating chart..."
- Log to stdout (scripts run in terminal) not to application logger

### 4.14 Reproducibility

**Requirements**:
- Save full experiment config as JSON alongside results (OutputConfig.save_config_json)
- Optional: capture git commit hash and dirty flag
- Optional: capture Python version, package versions
- Not required: seed control (our simulators are deterministic already — no randomness)

### 4.15 Integration & Migration Path

Cannot break 21 existing scripts at once.

**Requirements**:
- Phase 1: create `libs/backtest` alongside existing code. Both coexist.
- Phase 2: migrate one research script as a pilot (e.g. `v16a_universe_robustness.py`)
- Phase 3: migrate remaining scripts incrementally, prioritizing those still actively used
- Backward compat: `portfolio_backtest.py` continues to work; executor imports from it during transition
- Archive scripts that haven't been used in >2 months instead of migrating them

---

## 5. Variable Catalog

Covered in detail in `.tmp-backtest-vars.md`. ~70 configurable variables across 7 categories.

Key takeaway for infra design: the experiment config dataclasses above must be able to express any combination of these 70 variables without the user needing to understand all 70. Sensible defaults for everything; only override what changes.

---

## 6. CONSTITUTION Review

### ✅ Aligned

| Principle | How |
|---|---|
| **Less is more** | libs/backtest is minimal — 6 modules, no new services, no new dependencies |
| **High cohesion, low coupling** | Each module has one clear concern; chart module knows nothing about strategies |
| **Pragmatism and iteration** | Phase 1 ships the core then iterate; backward compat; pilot migration |
| **Research before action** | This document is the research. Variables inventoried. Conventions documented. |
| **First principles** | Questioned "should we use a framework?" → answered no. Built the right thing for our actual needs. |
| **Programming design** | Composition over inheritance — ChartSeries, PanelSpec are composed into figures; experiment configs are composed into runs |
| **Documentation design** | This is a single coherent doc, not scattered notes. Stable anchors, not fragile procedures. |

### ⚠️ Potential Concerns

| Concern | Mitigation |
|---|---|
| **Scope creep** — experiment config + sweep runner + caching is growing | Phase staging. Cache in Phase 2. Sweep runner only if scripts actually need it. |
| **Over-engineering** — is `ExperimentSweep` too "framework-y"? | If only 3-4 scripts use it, drop it. If 15+ scripts are essentially manual sweeps, it pays for itself. Defer decision. |
| **Backward compat breakage** — moving `portfolio_backtest.py` changes imports | Keep re-exports in executor during transition. Remove after all consumers migrated. |

### 🔴 One Violation to Address

The metrics module currently duplicates computation across three places (`core/metrics.py`, `portfolio_backtest.py`, ad-hoc in scripts). This violates **high cohesion** — three sources of truth for the same thing. The `BacktestMetrics` dataclass and `compute_metrics()` function directly fix this.

---

## 7. Implementation Phasing

### Phase 1 — Core Library (first PR)

- `libs/backtest/` workspace member with `pyproject.toml`
- `engine.py` — move simulators from `portfolio_backtest.py`
- `result.py` — `BacktestResult`, `BacktestMetrics`, `ChartSeries` dataclasses
- `metrics.py` — unified `compute_metrics()` covering Tier 1
- `data.py` — `DataConfig`, unified loading helper
- `chart.py` — `create_comparison_figure()`, `save_figure()`
- `experiment.py` — `BacktestExperiment` (single-run, no sweep yet)
- Keep `portfolio_backtest.py` as re-export shim for backward compat

### Phase 2 — Integration & Polish

- Migrate 1-2 research scripts as pilot
- Cache layer for target-set computation
- `ExperimentSweep` if justified by actual usage
- Additional Tier 2 metrics (Sterling, skewness, kurtosis)

### Phase 3 — Full Migration

- Migrate remaining active scripts
- Archive unused scripts
- Remove backward compat shims from executor
- Tier 3+ metrics as needed

---

## 8. Open Questions (for Boss)

1. **OK to drop `ExperimentSweep`** from Phase 1? Can defer until we see if multiple scripts actually need it. The individual experiment runner alone already eliminates 80% of boilerplate.

2. **Cache layer priority**: Is 2-5 minute backtest runtime acceptable per run? If yes, caching can wait. If we frequently do parameter sweeps where each run takes 5 minutes × 20 combos = 100 minutes, caching the target-set stage becomes valuable.

3. **Symbol universe naming**: Use `baseline_19`, `top10`, `majors_5` as standard names? Or something else?

4. **Backward compat timeline**: How long to keep `portfolio_backtest.py` as a re-export shim? One sprint? Until all scripts migrated?
