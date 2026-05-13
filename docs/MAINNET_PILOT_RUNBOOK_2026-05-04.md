# Mainnet pilot runbook — 2026-05-04

Purpose: define the public safety posture for the guarded mainnet pilot without
publishing private operator details.

Private deployment records must contain the live host, SSH user, exact account
state, capital, symbol universe, notification endpoints, active phase, order-cap
posture, and historical runtime snapshots. Do not duplicate those details in
this repository.

## Guardrail design

Mainnet v16a uses a dedicated guarded profile rather than the testnet profile.
Live order submission is fail-closed unless the expected network, profile, dry-run
state, and explicit allow flags are all present.

Risk limits are enforced in both configuration and code. Any change to live
phase, leverage, gross exposure, equity bounds, order caps, or uncapped-order
behavior is a live-risk decision and should go through research, code review, CI,
deployment, and post-deploy observation.

## Preflight

Before deployment, run the strict mainnet preflight from the private deployment
environment. It should perform only read-only exchange checks and fail deployment
on configuration, account-state, target-construction, data-freshness, path, or
symbol-universe errors.

Preflight output can include account and position information. Keep detailed JSON
on the deployment host or in private operator notes; do not print it to public CI
logs.

## Deployment

Use the GitHub Actions deployment workflow for normal releases. Avoid ad hoc host
mutation except read-only diagnostics or emergency stop/rollback.

For each live deployment:

1. Confirm the intended commit has passed CI.
2. Confirm the production compose files provide structure only, not stale runtime
   values.
3. Confirm private deployment configuration matches the approved live-risk
   record, including phase, scale/gross cap, leverage, symbol universe, and
   order-cap posture.
4. Deploy once, then observe health and the next scheduled tick before further
   experiments or configuration changes.

## Post-deploy checks

After deployment, check at minimum:

- container is running and not repeatedly restarting;
- OOM/restart counters are normal;
- expected non-secret guard flags are present;
- runtime env matches the approved private live-risk record; do not rely on code
  defaults or historical research docs for active phase/cap state;
- strict preflight passes;
- recent logs contain no unexpected errors, rejects, stale-data warnings, or order
  placement anomalies.

Do not publish detailed account, position, target-weight, or balance output in CI
logs or public docs.

## Forward diagnostics

Forward-shadow diagnostics must be read-only and isolated from the live runtime.
They must not change the live container environment, restart the live engine, run
a live tick, refresh data unexpectedly, or submit orders.

Keep detailed snapshot journals private when they include live account state,
positions, target weights, balances, or host paths. Public research notes should
summarize only the method and conclusions.

## Stop / rollback

Stopping new orders and flattening positions are separate decisions. Emergency
procedures should live in private operator notes with the current host and account
context.
