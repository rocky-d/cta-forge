# Multi-Instance Deployment Incident — 2026-05-29 Retrospective

## Summary

A configuration mismatch caused the two live instances (mainnet-pilot and mainnet-400-01) to produce different strategy outputs for ~9 days. Root cause was `V16A_CORE_PHASE_HOURS=2` on pilot vs `=0` on 400-01.

## Root Cause

`V16A_CORE_PHASE_HOURS` controls the UTC-hour phase alignment of synthesized 6h bars in the v10g core sleeve. Phase 2 means bars align to 02:00/08:00/14:00/20:00 UTC; phase 0 means 00:00/06:00/12:00/18:00.

The 400-01 example env template had `V16A_CORE_PHASE_HOURS=0` (the code default). The pilot's production env had `V16A_CORE_PHASE_HOURS=2` (set during May 6 time-phase shadow diagnostics, carried into production via May 13 compose consolidation, never reverted because it was intentional — phase 2 showed better backtest performance).

Both values are valid; the problem was they didn't match. Same strategy code, different bar alignment, different signals.

## Timeline

| Date | Event |
|------|-------|
| 2026-05-06 | `V16A_CORE_PHASE_HOURS` parameter introduced (commit 049148d) |
| 2026-05-06~13 | Pilot's compose overlay set to 2 for shadow diagnostics |
| 2026-05-13 | Compose consolidation carried value 2 into production.env |
| 2026-05-21 | 400-01 created with example env using default value 0 |
| 2026-05-29 | Divergence discovered; 400-01 aligned to pilot's value 2 |

## Contributing Errors

1. **New instance env not diffed against production.** The example template was used as-is without comparing to the running pilot's env. Always `diff <(grep -v SECRET prod.env) <(grep -v SECRET new.env)` before starting a new instance.

2. **Premature conclusion about 6h data.** Pilot's 6h parquet files were observed stale (May 12) and initially flagged as a bug. With `CORE_PHASE_HOURS=2`, 6h bars are synthesized from 1h data — the files are intentionally not used. Verify code paths before declaring data issues.

3. **Mid-tick deploy.** A deploy triggered at :00 UTC restarted the pilot container while it was fetching data, causing missed tick (bar 602). Defer: deploy safety infrastructure now has CI gate (15-45 window), stop_grace_period 600s, and SIGTERM graceful shutdown.

4. **Spot/perp confusion.** Initially claimed HL funds needed internal transfer from spot to perp. HL is a unified account — spot USDC is directly available as perp margin. No transfer step exists.

## Prevention Checklist (new instance onboarding)

When creating a new live instance:
- [ ] Diff ALL env vars against an existing production instance
- [ ] Any difference must be intentional and documented
- [ ] Shared config extraction: params that should always match go in a shared config
- [ ] Run preflight BEFORE starting
- [ ] Start in DRY_RUN mode, observe at least 24h of matching targets before promoting

## Key Architectural Facts (verified, don't repeat mistakes)

- **HL account**: Unified spot+perp. USDC in spot = immediately available margin. No transfer needed.
- **Journal persistence**: Both instances use `PERSISTENCE_BACKEND=postgres`. File journals are dead artifacts from pre-2026-05-20 era.
- **V16a CORE_PHASE_HOURS**: Controls 6h bar phase alignment. Both instances must use the same value.
- **6h parquet files**: With CORE_PHASE_HOURS≠0, 6h bars are synthesized from 1h, not loaded from parquet. 6h parquet files are not needed.
- **ParquetStore.write() is NOT atomic**: Cannot share data directory between concurrent processes.
- **Binance rate limits**: Both instances fetching 1h data simultaneously causes ~120 retries per tick. Acceptable for 1h candles (~2min extra latency). Long-term: independent data-fetching service.
