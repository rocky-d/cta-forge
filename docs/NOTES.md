# cta-forge — Project Notes

Deeper technical reference supplementing [README.md](../README.md) (canonical overview).

Crypto CTA trend-following strategy engine. Monorepo with 5 microservices + 2 shared libraries.
Market data primarily from Binance USDS-M futures; live execution on Hyperliquid mainnet.

## Strategy

Current production profile: **v16a Badscore Overlay** (`v16a-mainnet-pilot`).

### v10g engine (core sleeve)

Multi-factor trend-following:

- Multi-timeframe momentum (20/60/120 bars)
- Ensemble ADX filter (22/27/32)
- Donchian breakout (20-bar channel)
- Volume confirmation
- DI directional filter
- BTC correlation dampening
- Signal persistence (2 consecutive bars)

Position management: risk parity sizing, 12% annualized vol target, ATR trailing stops (5.0→3.0x), max hold 100 bars, min hold 16 bars, partial take-profit at 2.5x ATR, max 5 concurrent positions, 8% drawdown breaker.

### v16a overlay

Research checkpoint: [`STRATEGY_ITERATION_2026-04-28.md`](STRATEGY_ITERATION_2026-04-28.md).

- Core sleeve: shifted v10g 6h trend strategy.
- Overlay sleeve: 1h fast-exit top-2 signals.
- Regime risk gate: badscore (vol + trend efficiency + cross-asset correlation).
- Allocation: 50% core / 50% overlay.

## Design Principles

- Protocol + composition over ABC/inheritance
- No: ccxt, loguru, pandas (prefer polars/numpy), LangChain
- Ruff lint + format, ty type check, pytest
- Factor auto-discovery via registry
- Parquet storage: `{data_dir}/{symbol}/{interval}.parquet` (zstd)

## Live Trading

Production runs on EC2 (Tokyo, t3.small) with Docker Compose.

- **Profile**: `v16a-mainnet-pilot`, 19-symbol universe, Hyperliquid mainnet
- **Persistence**: PostgreSQL primary (`PERSISTENCE_BACKEND=postgres`), Docker volumes for parquet cache
- **Deploy**: GitHub Actions workflow_dispatch → GHCR → SSH EC2
- **Guard**: preflight checks before every deploy; multi-layer allow flags (strategy profile, network, instance-specific)
- **Notifications**: Telegram (main), Lark/Feishu (secondary)

Second dry-run instance (`mainnet-400-01`) is prepared via the `mainnet-400-01` Compose profile. See [`MULTI_LIVE_MINIMAL_ROLLOUT_2026-05-21.md`](MULTI_LIVE_MINIMAL_ROLLOUT_2026-05-21.md).

Key design invariants:

- `V10GDecisionEngine.tick()` mutates internal state. Callers needing pre-tick positions must snapshot before calling `tick()`.
- Drawdown is a positive magnitude from peak in journals; only charts may negate it for display.
- All live-risk changes (phase, leverage, cap, symbols) must come from verified private env, not code defaults or old research docs.

## Backtest

Backtest entry points are thin CLIs over executor modules:

```bash
uv run python scripts/backtest/v10g_maxrange.py        # v10g
uv run python scripts/backtest/joint_badscore_research.py  # v16a
```

Chart convention (Boss-facing): 16:9 PNG, 3 panels (equity, drawdown, monthly P&L), max 4 configs per chart.

## Where to find more

| Topic | Document |
|-------|----------|
| Mainnet pilot runbook & guards | [`MAINNET_PILOT_RUNBOOK_2026-05-04.md`](MAINNET_PILOT_RUNBOOK_2026-05-04.md) |
| Multi-live (400-01) rollout | [`MULTI_LIVE_MINIMAL_ROLLOUT_2026-05-21.md`](MULTI_LIVE_MINIMAL_ROLLOUT_2026-05-21.md) |
| Strategy iteration research | [`STRATEGY_ITERATION_2026-04-28.md`](STRATEGY_ITERATION_2026-04-28.md) |
| Time-phase research | [`TIME_PHASE_RESEARCH_2026-05-06.md`](TIME_PHASE_RESEARCH_2026-05-06.md) |
| DBMS/multi-live design | [`DBMS_MULTI_LIVE_DESIGN_2026-05-14.md`](DBMS_MULTI_LIVE_DESIGN_2026-05-14.md) |
| DBMS dual-write design | [`DBMS_DUAL_WRITE_DESIGN_2026-05-15.md`](DBMS_DUAL_WRITE_DESIGN_2026-05-15.md) |
| DBMS cutover checklist | [`DBMS_CUTOVER_CHECKLIST_2026-05-15.md`](DBMS_CUTOVER_CHECKLIST_2026-05-15.md) |
| DBMS backup/restore | [`DBMS_BACKUP_RESTORE_RUNBOOK_2026-05-15.md`](DBMS_BACKUP_RESTORE_RUNBOOK_2026-05-15.md) |
| DBMS container deployment | [`DBMS_CONTAINER_DEPLOYMENT_2026-05-15.md`](DBMS_CONTAINER_DEPLOYMENT_2026-05-15.md) |
| HL mainnet/testnet research | [`HL_MAINNET_TESTNET_RESEARCH_2026-05-02.md`](HL_MAINNET_TESTNET_RESEARCH_2026-05-02.md) |
| HL SDK known issues | [`HYPERLIQUID_SDK_ISSUES.md`](HYPERLIQUID_SDK_ISSUES.md) |
| Historical import plan | [`LIVE_PERSISTENCE_IMPORT_PLAN_2026-05-15.md`](LIVE_PERSISTENCE_IMPORT_PLAN_2026-05-15.md) |
| Import review | [`LIVE_PERSISTENCE_IMPORT_REVIEW_2026-05-16.md`](LIVE_PERSISTENCE_IMPORT_REVIEW_2026-05-16.md) |
| Multi-live config incident (2026-05-29) | [`INCIDENT_2026-05-29_MULTI_LIVE_CONFIG.md`](INCIDENT_2026-05-29_MULTI_LIVE_CONFIG.md) |
