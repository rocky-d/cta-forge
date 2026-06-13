# Env Variable Cleanup — Tracking

Started: 2026-06-13 14:56 UTC
Context: Full audit of all executor env vars + code references → 6 findings

## Issues & Status

### 1. `ALLOW_MAINNET_PILOT_LIVE` stale ref in routes.py
- [ ] Code: remove from `services/executor/src/executor/routes.py:145-147`
- [ ] Commit + push
- [ ] CI: executor image rebuild
- [ ] Deploy: restart executor containers on EC2

### 2. `MAX_ORDER_NOTIONAL` empty — add clarifying comment
- [ ] Env template: `infra/env.mainnet-400-01.example` — add comment
- [ ] Production: pilot `.env` on EC2 — add comment
- [ ] Production: 400-01 `.env.mainnet-400-01` on EC2 — add comment

### 3. `MAX_EQUITY` vs `MAINNET_MAX_EQUITY` naming — add clarifying comment
- [ ] Env template: `infra/env.mainnet-400-01.example` — add section comment
- [ ] Production: pilot `.env` on EC2 — add comment
- [ ] Production: 400-01 `.env.mainnet-400-01` on EC2 — add comment

### 4. `TARGET_GROSS_CAP=4.00` precision
- [x] Decision: cosmetic only, no action needed

### 5. `SHADOW_WRITE_FAILURE_POLICY` legacy fallback
- [x] Decision: intentional backward compat, no action needed

### 6. `ALLOW_V16A_TESTNET_LIVE` — no issue
- [x] Decision: testnet-only var, correctly unset for mainnet

---

## Execution Order

Phase A — Code fix (safe, offline):
  1. Remove ALLOW_MAINNET_PILOT_LIVE from routes.py
  2. Commit + push → CI builds new executor image

Phase B — Env template (safe, offline):
  3. Update env.mainnet-400-01.example with clarifying comments

Phase C — Production env (safe, no restart needed):
  4. Update pilot .env on EC2 (comments only)
  5. Update 400-01 .env.mainnet-400-01 on EC2 (comments only)

Phase D — Deploy (requires container restart):
  6. Pull new executor image on EC2
  7. Restart executor containers (between ticks, ~5s downtime)
  8. Verify both instances resume normally

## Safety Notes
- Phase A+B+C are zero-risk (comments + API display field)
- Phase D: routes.py change is display-only, no trading logic affected
- Only executor containers need restart (not postgres, not dashboard)
- Restart between hourly ticks to avoid interrupting in-flight orders
