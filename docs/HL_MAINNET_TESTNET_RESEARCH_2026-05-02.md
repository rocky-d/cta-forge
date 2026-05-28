# Hyperliquid SDK Mainnet/Testnet Difference Research

_Date: 2026-05-02_

## Scope

Research focused on whether `cta-forge` can infer mainnet safety from current Hyperliquid testnet-live behavior, especially at SDK/API integration level.

Sources checked:

- Hyperliquid official API docs: exchange endpoint / signing fields.
- Official `hyperliquid-python-sdk` repository source and open issues/PRs.
- Current local adapter and live runner code.
- Public `/info` endpoint metadata and L2 snapshots for both mainnet and testnet.

## Current project state

- `hyperliquid-python-sdk==0.23.0` is locked in `uv.lock`; PyPI latest at check time is also `0.23.0`.
- Adapter chooses `constants.TESTNET_API_URL` or `constants.MAINNET_API_URL` from the SDK.
- v16a non-dry-run mainnet was initially blocked in `services/executor/src/executor/run_live.py`; this was intentional at the time of writing (2026-05-02). Mainnet is now live with the `v16a-mainnet-pilot` profile since 2026-05-05.
- Testnet filters out `XRP`, `LINK`, `DOT`, `SEI` because these exist on mainnet but are unavailable on Hyperliquid testnet.

## Findings

### 1. Endpoint/signing split is real but basic L1 orders are handled correctly

Official SDK constants define:

- Mainnet: `https://api.hyperliquid.xyz`
- Testnet: `https://api.hyperliquid-testnet.xyz`

SDK order placement signs L1 actions with `is_mainnet = self.base_url == MAINNET_API_URL`. Our adapter passes the SDK constant URL exactly, so basic perp order signatures should use the correct mainnet/testnet source.

The official docs also distinguish `hyperliquidChain: Mainnet` vs `Testnet` and `signatureChainId` for user-signed actions. Current SDK master still hardcodes `signatureChainId = 0x66eee` in `sign_user_signed_action`; issue #254 and PR #267 discuss this. This mainly affects user-signed transfer/abstraction actions, not our normal `Exchange.order` path, which uses `sign_l1_action`.

### 2. Testnet metadata has had real inconsistencies

Official SDK issues #273/#275 report testnet-only `spotMeta` invalid token indices causing `Info` initialization failures. This matches our earlier local finding. SDK 0.23.0 changed token lookup behavior, and our adapter additionally filters malformed spot pairs defensively before constructing `Info` / `Exchange`.

Conclusion: testnet data quality is weaker than mainnet, and testnet success does not prove all mainnet metadata/universe behavior.

### 3. Mainnet/testnet universes differ

Public metadata check on 2026-05-02:

- Mainnet perp universe size: 230
- Testnet perp universe size: 208
- Mainnet includes `XRP`, `LINK`, `DOT`, `SEI`; testnet misses at least `XRP`, `SEI` from the v10g/v16 symbol set, and code already excludes the known testnet-only-missing set.
- Some symbol metadata differs across networks. Examples:
  - `SOL` max leverage: mainnet 20, testnet 10
  - `APT` max leverage: mainnet 10, testnet 3
  - `OP` max leverage: mainnet 5, testnet 50
  - Asset indices/order differ substantially between networks.

Because the SDK maps by name to network-local asset id, different indices are okay mechanically. But any static assumptions about universe ordering would be unsafe.

### 4. L2/liquidity behavior is very different

One public L2 snapshot on 2026-05-02 showed testnet spreads were generally wider/thinner than mainnet for many symbols. Example spread bps from that instant:

- BTC: mainnet ~0.13 bps, testnet ~2.04 bps
- ETH: mainnet ~0.43 bps, testnet ~25.90 bps
- SUI: mainnet ~0.11 bps, testnet ~19.71 bps
- INJ: mainnet ~4.23 bps, testnet ~7.66 bps

For 100-200 USDC, mainnet liquidity is likely better than testnet for large caps. But small account constraints still matter: minimum notional, size decimals, fees, IOC slippage, and signal churn.

## Risk interpretation for mainnet rollout

The biggest SDK-level concerns are not “mainnet will be worse than testnet in every way.” They are:

1. Testnet has different universe and metadata quality, so it is not a perfect production mirror.
2. User-signed action handling has open upstream discussion around chain id, though our normal order path is not the affected path.
3. Mainnet unlocks symbols testnet excluded, so first mainnet targets may include symbols that were not exercised in testnet-live.
4. Public market microstructure differs materially; testnet order-flow success is not the same as mainnet execution economics.
5. Current v16a mainnet was intentionally code-blocked at the time of writing; it was enabled with an explicit guardrail change on 2026-05-05 under the `v16a-mainnet-pilot` profile.

## Recommended investigation before mainnet live

1. Add/read-only mainnet dry-run probe using a mainnet account with no trading:
   - initialize adapter against mainnet;
   - fetch `meta`, `spotMeta`, account state, open orders;
   - fetch L2 snapshots for the configured symbols;
   - compute current v16a target weights with mainnet universe, but do not place orders.

2. Add a mainnet symbol eligibility report:
   - symbol exists;
   - `szDecimals`;
   - max leverage;
   - latest best bid/ask;
   - estimated minimum rounded order size for 100/200 USDC equity.

3. ✅ Completed 2026-05-05. Mainnet pilot is live.

4. If proceeding, use a separate tiny mainnet config/profile with:
   - no leverage;
   - total equity cap 100-200 USDC;
   - conservative `MIN_ORDER_NOTIONAL`;
   - reduced symbol universe at first, preferably only the most liquid symbols;
   - kill switch / rollback command documented.

## Source links

- Hyperliquid exchange endpoint docs: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/exchange-endpoint
- SDK constants: https://github.com/hyperliquid-dex/hyperliquid-python-sdk/blob/master/hyperliquid/utils/constants.py
- SDK signing: https://github.com/hyperliquid-dex/hyperliquid-python-sdk/blob/master/hyperliquid/utils/signing.py
- SDK exchange order path: https://github.com/hyperliquid-dex/hyperliquid-python-sdk/blob/master/hyperliquid/exchange.py
- Issue #273: https://github.com/hyperliquid-dex/hyperliquid-python-sdk/issues/273
- Issue #275: https://github.com/hyperliquid-dex/hyperliquid-python-sdk/issues/275
- Issue #254: https://github.com/hyperliquid-dex/hyperliquid-python-sdk/issues/254
- PR #267: https://github.com/hyperliquid-dex/hyperliquid-python-sdk/pull/267
