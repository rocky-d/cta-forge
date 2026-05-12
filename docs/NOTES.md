# cta-forge -- NOTES.md

## Project Overview

Crypto CTA (Commodity Trading Advisor) trend-following strategy engine.
Monorepo with 5 microservices + 2 shared libraries. Historical/cache data
currently comes primarily from Binance USDS-M futures, while live execution
targets Hyperliquid.

## Architecture

```
libs/
  core/              -- shared protocols, constants, metrics
  exchange/          -- exchange connectivity (Hyperliquid adapter)
services/
  data-service/      -- Binance kline fetcher, parquet store, REST API
  alpha-service/     -- factor computation (TSMOM, Donchian, VolRegime, V10GComposite)
  strategy-service/  -- signal composition, allocation, risk
  executor/          -- backtest engine, target-weight portfolio layer & live execution
  report-service/    -- metrics calculation, chart generation (3-panel backtest chart)
scripts/backtest/    -- CLI entry point (thin wrapper, imports from services)
backtest-results/    -- output charts, metrics JSON
docs/                -- project documentation
```

Ports and service URLs have sensible defaults in `core/constants.py`, overridable via environment variables.

## Strategy: v10g

Multi-factor trend-following with adaptive features.

Research checkpoint: the current best robust iteration candidate is documented in
[`STRATEGY_ITERATION_2026-04-28.md`](STRATEGY_ITERATION_2026-04-28.md). It now
has a reusable target-weight profile/backtest path. The code default remains
v10g, while the production testnet compose can explicitly promote v16a with
separate guard flags.

Signals:
- Multi-timeframe momentum (lookbacks: 20/60/120 bars)
- Ensemble ADX filter (thresholds: 22/27/32, averaged)
- Donchian breakout confirmation (20-bar channel)
- Volume confirmation (20-bar avg ratio)
- DI directional filter (+DI/-DI)
- BTC correlation filter (dampen alt signals contradicting BTC trend)
- Adaptive lookback weighting (high vol favors short lookback)
- Signal persistence (require 2 consecutive bars same direction)

Position management:
- Risk parity sizing (inverse vol contribution)
- Vol targeting (12% annualized, EMA of recent returns)
- ATR trailing stop (5.0x, tightens to 3.0x after 2.0x ATR profit)
- Max hold 100 bars (forced exit)
- Min hold 16 bars (prevent whipsaw)
- Partial take-profit at 2.5x ATR
- Max 5 concurrent positions
- Rebalance every 4 bars
- 8% drawdown circuit breaker (halves new position size)

## Design Principles

- Protocol + composition over ABC/inheritance
- No: ccxt, loguru, pandas (use polars/numpy), LangChain
- Ruff lint + format, ty type check, pytest
- Factor auto-discovery via registry
- Parquet storage: `{data_dir}/{symbol}/{interval}.parquet` (zstd)
- Signal range: normalized to [-1, +1]

## Live Trading (Hyperliquid)

Architecture:
- `libs/exchange/` -- HL SDK wrapper library
  - `ExchangeAdapter` Protocol interface (structural subtyping)
  - `HyperliquidAdapter` implementation (SDK 0.23 metadata-safe init, unified account, async executor)
- `services/executor/src/executor/decision.py` -- V10GDecisionEngine (stateless decision logic, shared by live + backtest)
- `services/executor/src/executor/targeting.py` -- target-weight portfolio abstractions and target-to-order delta calculation
- `services/executor/src/executor/portfolio_backtest.py` -- target-weight portfolio simulation utilities
- `services/executor/src/executor/profiles/v16a_badscore_overlay.py` -- reusable v16a Badscore Overlay target profile
- `services/executor/src/executor/live.py` -- LiveEngine (code default: v10g strategy, 6h candle-aligned loop; target-mode reconciliation is available behind explicit strategy injection and is used by the guarded v16a testnet-live deployment)
- `services/executor/src/executor/notify.py` -- Notifier stack (Telegram, Lark/Feishu, multi-backend)
- `services/executor/src/executor/run_live.py` -- CLI entry point

Preflight checks (must all pass before trading):
1. Exchange connectivity
2. Minimum equity ($100)
3. Position reconcile (state vs exchange, adopt orphaned positions)
4. Stale open orders -> auto-cancel
5. Market data spot check

Risk controls:
- Drawdown is represented project-wide as a positive magnitude from peak (for example 5% below peak -> `0.05`, journal `dd_pct=5.0`). Only chart rendering may negate it visually for underwater plots.
- 15% max drawdown -> hard stop (flatten all)
- 8% drawdown -> DD breaker (50% position size reduction)
- ATR trailing stops (5.0x, tightens to 3.0x after 2.0x ATR profit)
- Max 5 concurrent positions
- 15% max per-position equity

Important: V10GDecisionEngine.tick() mutates state internally (deletes closed
positions, adjusts partial qty, updates best_price). Callers must snapshot
positions before tick() if they need pre-tick state for settlement.

Ops:
- Env vars: HL_PRIVATE_KEY, HL_ACCOUNT_ADDRESS, HL_NETWORK, DRY_RUN, TG_BOT_TOKEN, TG_CHAT_ID, LARK_WEBHOOK_URL, DATA_DIR, JOURNAL_DIR, STRATEGY_PROFILE, MIN_ORDER_NOTIONAL, MAX_ORDER_NOTIONAL, MIN_EQUITY, MIN_AVAILABLE_BALANCE, MAX_EQUITY, TARGET_SCALE, TARGET_GROSS_CAP, HL_LEVERAGE, LIVE_SYMBOLS, V16A_MAX_STALENESS_HOURS, V16A_CORE_PHASE_HOURS, V16A_COMPARE_CORE_PHASE_HOURS, PHASE_COMPARISON_JOURNAL_DIR, ALLOW_V16A_TESTNET_LIVE, ALLOW_MAINNET_PILOT_LIVE, ALLOW_MAINNET_PILOT_UNCAPPED_ORDERS
- `STRATEGY_PROFILE` code default is `v10g-engine-6h`. `v16a-badscore-overlay` is shadow-safe by default; non-dry-run v16a requires both `HL_NETWORK=testnet` and explicit `ALLOW_V16A_TESTNET_LIVE=true`. Mainnet uses the separate guarded `v16a-mainnet-pilot` profile. Keep active mainnet pilot sizing in `MAINNET_PILOT_RUNBOOK_2026-05-04.md` and compose overlays, not duplicated here.
- One-shot v16a shadow command: `DRY_RUN=true STRATEGY_PROFILE=v16a-badscore-overlay uv run python -m executor.run_shadow_tick`. Run it locally, or as an explicit one-off executor command only after review; do not use the deploy workflow for experiments. The command requires `HL_PRIVATE_KEY` and `HL_ACCOUNT_ADDRESS` in the environment so it can read account/position state, but it rejects `DRY_RUN=false`.
- v16a target freshness: the latest 6h v10g core bar is forward-filled across the following live 6h window while the 1h overlay can update hourly. Do not forward-fill past the next 6h bar close unless a newer core target exists.
- v16a core phase config: `V16A_CORE_PHASE_HOURS` defaults to `0` and accepts integer UTC-hour offsets `0..5`. Non-zero values build the 6h v10g core sleeve from refreshed 1h cache while leaving the 1h overlay unchanged. Keep `0` for live defaults until time-phase research is promoted.
- v16a phase comparison shadow: set `V16A_COMPARE_CORE_PHASE_HOURS=2` on `executor.run_shadow_tick` to write read-only side-by-side phase diagnostics to `PHASE_COMPARISON_JOURNAL_DIR` (default `journal/phase-shadow`). This records target diff metrics and hypothetical orders; it must remain dry-run/no-order.
- Cache-only phase shadow: `DRY_RUN=true STRATEGY_PROFILE=v16a-badscore-overlay V16A_COMPARE_CORE_PHASE_HOURS=2 uv run python -m executor.run_phase_shadow_snapshot` records the same side-by-side phase diagnostics without running a `LiveEngine` tick and without refreshing Binance data. Use this for forward evidence collection when the live cache is already maintained elsewhere.
- v16a live-like constraint ablation: `uv run python scripts/backtest/v16a_live_constraints_ablation.py` compares ideal target-weight results with `$100` pilot equity, `$10` min order, `$50` max increase order, scale/gross-cap variants, and slippage assumptions. Current finding: the fixed max-increase cap dominates the gap from theory; keep live unchanged and treat a possible `$75` cap as a shadow/research candidate, not an immediate live change.
- v16a mainnet-pilot forward shadow: after deploying commit `47fdea0` on 2026-05-06, the first cache-only EC2 snapshots used `DRY_RUN=true STRATEGY_PROFILE=v16a-badscore-overlay V16A_COMPARE_CORE_PHASE_HOURS=2` only in the snapshot subprocess while live stayed `v16a-mainnet-pilot`, phase `0`, cap `$50`. The 14:00 UTC target snapshot showed phase0-vs-phase2 L1 `0.39148245`, cosine `0.90401641`, no sign flips, and no ignored gross. Phase 0 generated 0 hypothetical orders; phase 2 generated two reduce-only sells (`SEI`, `XRP`). `$50` and `$75` caps were identical in this point because reduce-only orders are uncapped.
- v16a live target construction is cache-only: `LiveEngine` refreshes required parquet data asynchronously before target generation, and `V16aOnlineTargetStrategy` must not fetch/backfill or start an event loop inside synchronous target calculation.
- v16a staleness guard: `V16A_MAX_STALENESS_HOURS` defaults to 8 for shadow and guarded testnet-live target generation. If ticks show targets older than the expected 1h overlay cadence outside data/API outages, investigate before further rollout or any mainnet discussion.
- State persistence: `engine-state.json` for live mode and `engine-state-shadow.json` for shadow mode (auto-generated, gitignored)
- Journal outputs: `equity.jsonl`, `trades.jsonl`, `signals.jsonl`, and target-mode `targets.jsonl` diagnostics. Target diagnostics include staleness, execution coverage, and ignored gross so testnet/mainnet universe gaps stay visible.
- Data cache: parquet files in DATA_DIR (live + backtest share via ParquetStore)
- Deployment: GitHub Actions workflow_dispatch -> GHCR -> SSH EC2 (Tokyo t3.small). Prefer this CI/CD path for v16a promotion; avoid ad hoc EC2 changes except read-only checks or urgent diagnostics. Deploy target `testnet-live` uses only `docker-compose.prod.yml`; target `mainnet-pilot-dry-run` overlays `docker-compose.mainnet-pilot.yml`; target `mainnet-pilot-live` additionally overlays `docker-compose.mainnet-pilot-live.yml`. Secrets and notification endpoints remain in the EC2 `.env`.

## CI/CD

- Lint: ruff check + ruff format + ty check (3 parallel jobs)
- Test: pytest (unit + integration)
- Deploy: workflow_dispatch -> check-ci gate -> build Docker -> push GHCR -> SSH deploy EC2
- Workflow action runtime versions are owned by the referenced actions; watch GitHub annotations for upstream deprecation warnings.

## Backtest

Architecture:
- `executor` is the source of truth for strategy and backtest logic.
- `scripts/backtest/*.py` are intentionally thin reproduction CLIs: they choose a profile, call executor modules, call report-service plotting, and write local artifacts.
- `services/executor/src/executor/signal_pipeline.py` -- shared historical data and signal pipeline
  - `fetch_bars()` -- ParquetStore cache + Binance incremental fetch
  - `precompute()`, `build_timeline()`, `align_data()`, `compute_signals()` -- shared data pipeline
- `services/executor/src/executor/backtest.py` -- V10GDecisionEngine action-based backtest module
  - `run_backtest()` -- V10GDecisionEngine loop (same engine as live trading)
  - `run_full_backtest()` -- single orchestration entry point over `signal_pipeline`
- `services/executor/src/executor/profiles/v16a_badscore_overlay.py` -- reusable v16a profile and target construction logic, also using `signal_pipeline`
- `services/executor/src/executor/portfolio_backtest.py` -- target-weight and simple execution-realistic portfolio backtest paths used by v16a
- `services/report-service/src/report_service/plot.py` -- `plot_backtest()` three-panel chart
  (equity + BTC/ETH indexed overlay, drawdown, monthly returns)

Usage:
- v10g CLI: `uv run python scripts/backtest/v10g_maxrange.py` (thin wrapper around `executor.backtest`)
- v16a CLI: `uv run python scripts/backtest/joint_badscore_research.py` (thin wrapper around `executor.profiles.v16a_badscore_overlay` + `executor.portfolio_backtest`)
- REST API:
  1. `POST :8004/backtest` -> v10g JSON (metrics, equity_curve, btc/eth prices, yearly)
  2. `POST :8005/plot/backtest` -> PNG (three-panel chart with BTC/ETH overlay)

Output examples: `backtest-results/backtest_v10g_6h_default.png`, `backtest-results/backtest_joint_badscore_research.png`, and matching `metrics_*.json` files.
