
  2026-05-19T02:03:37Z  target_sell  AVAX dash=1.0866941783726713 → hl=1.09 (diff=+0.003306)
  2026-05-18T20:03:34Z  target_sell   SOL dash=0.23518226711193901 → hl=0.24 (diff=+0.004818)
  2026-05-18T13:03:37Z   target_buy    OP dash=88.46676162740549 → hl=88.5 (diff=+0.033238)
  2026-05-18T06:03:33Z   target_buy   ARB dash=277.1557787194071 → hl=277.2 (diff=+0.044221)
  2026-05-16T13:03:33Z  target_sell    OP dash=456.2523044735708 → hl=456.3 (diff=+0.047696)
  2026-05-16T09:03:32Z  target_sell   ARB dash=560.3595441780957 → hl=560.4 (diff=+0.040456)
  2026-05-14T01:03:31Z  target_sell   INJ dash=3.8554149848318757 → hl=3.9 (diff=+0.044585)
  2026-05-13T20:03:30Z   target_buy   BNB dash=0.021590084337519723 → hl=0.022 (diff=+0.000410)
  2026-05-11T03:03:29Z  target_sell   SEI dash=186.9726623153363 → hl=187.0 (diff=+0.027338)
  2026-05-10T17:03:30Z  target_sell   SEI dash=192.87325429316707 → hl=193.0 (diff=+0.126746)
  2026-05-08T21:03:32Z   target_buy   TIA dash=26.36154012 → hl=26.4 (diff=+0.038460)
  2026-05-07T01:03:31Z   target_buy   SOL dash=0.3965299 → hl=0.4 (diff=+0.003470)
  2026-05-07T01:03:31Z   target_buy   XRP dash=8.93226977 → hl=9.0 (diff=+0.067730)
  2026-05-06T13:03:33Z   target_buy   SOL dash=0.41500174 → hl=0.42 (diff=+0.004998)
  2026-05-05T20:00:42Z   target_buy   SEI dash=260.73659939 → hl=261.0 (diff=+0.263401)
  2026-05-05T20:00:41Z   target_buy  LINK dash=1.51916293 → hl=1.5 (diff=-0.019163)
  2026-05-05T19:00:43Z   target_buy  LINK dash=2.6492713 → hl=2.6 (diff=-0.049271)
  2026-05-05T19:00:42Z   target_buy   INJ dash=3.17404151 → hl=3.2 (diff=+0.025958)

-- Fix historical trade qty in live_trades
-- Generated at 2026-05-28T03:13:34.641359+00:00
-- 18 rows to update
BEGIN;

UPDATE live_trades SET qty = 1.09 WHERE ts::text LIKE '2026-05-19T02:03:37%'   AND symbol = 'AVAX'   AND kind IN ('target_buy', 'target_sell', 'buy', 'sell')   AND ABS(qty - 1.0866941783726713) < 0.01;  -- #1: 2026-05-19T02:03:37Z target_sell AVAX 1.0866941783726713 → 1.09
UPDATE live_trades SET qty = 0.24 WHERE ts::text LIKE '2026-05-18T20:03:34%'   AND symbol = 'SOL'   AND kind IN ('target_buy', 'target_sell', 'buy', 'sell')   AND ABS(qty - 0.23518226711193901) < 0.01;  -- #2: 2026-05-18T20:03:34Z target_sell SOL 0.23518226711193901 → 0.24
UPDATE live_trades SET qty = 88.5 WHERE ts::text LIKE '2026-05-18T13:03:37%'   AND symbol = 'OP'   AND kind IN ('target_buy', 'target_sell', 'buy', 'sell')   AND ABS(qty - 88.46676162740549) < 0.01;  -- #3: 2026-05-18T13:03:37Z target_buy OP 88.46676162740549 → 88.5
UPDATE live_trades SET qty = 277.2 WHERE ts::text LIKE '2026-05-18T06:03:33%'   AND symbol = 'ARB'   AND kind IN ('target_buy', 'target_sell', 'buy', 'sell')   AND ABS(qty - 277.1557787194071) < 0.01;  -- #4: 2026-05-18T06:03:33Z target_buy ARB 277.1557787194071 → 277.2
UPDATE live_trades SET qty = 456.3 WHERE ts::text LIKE '2026-05-16T13:03:33%'   AND symbol = 'OP'   AND kind IN ('target_buy', 'target_sell', 'buy', 'sell')   AND ABS(qty - 456.2523044735708) < 0.01;  -- #5: 2026-05-16T13:03:33Z target_sell OP 456.2523044735708 → 456.3
UPDATE live_trades SET qty = 560.4 WHERE ts::text LIKE '2026-05-16T09:03:32%'   AND symbol = 'ARB'   AND kind IN ('target_buy', 'target_sell', 'buy', 'sell')   AND ABS(qty - 560.3595441780957) < 0.01;  -- #6: 2026-05-16T09:03:32Z target_sell ARB 560.3595441780957 → 560.4
UPDATE live_trades SET qty = 3.9 WHERE ts::text LIKE '2026-05-14T01:03:31%'   AND symbol = 'INJ'   AND kind IN ('target_buy', 'target_sell', 'buy', 'sell')   AND ABS(qty - 3.8554149848318757) < 0.01;  -- #7: 2026-05-14T01:03:31Z target_sell INJ 3.8554149848318757 → 3.9
UPDATE live_trades SET qty = 0.022 WHERE ts::text LIKE '2026-05-13T20:03:30%'   AND symbol = 'BNB'   AND kind IN ('target_buy', 'target_sell', 'buy', 'sell')   AND ABS(qty - 0.021590084337519723) < 0.01;  -- #8: 2026-05-13T20:03:30Z target_buy BNB 0.021590084337519723 → 0.022
UPDATE live_trades SET qty = 187.0 WHERE ts::text LIKE '2026-05-11T03:03:29%'   AND symbol = 'SEI'   AND kind IN ('target_buy', 'target_sell', 'buy', 'sell')   AND ABS(qty - 186.9726623153363) < 0.01;  -- #9: 2026-05-11T03:03:29Z target_sell SEI 186.9726623153363 → 187.0
UPDATE live_trades SET qty = 193.0 WHERE ts::text LIKE '2026-05-10T17:03:30%'   AND symbol = 'SEI'   AND kind IN ('target_buy', 'target_sell', 'buy', 'sell')   AND ABS(qty - 192.87325429316707) < 0.01;  -- #10: 2026-05-10T17:03:30Z target_sell SEI 192.87325429316707 → 193.0
UPDATE live_trades SET qty = 26.4 WHERE ts::text LIKE '2026-05-08T21:03:32%'   AND symbol = 'TIA'   AND kind IN ('target_buy', 'target_sell', 'buy', 'sell')   AND ABS(qty - 26.36154012) < 0.01;  -- #11: 2026-05-08T21:03:32Z target_buy TIA 26.36154012 → 26.4
UPDATE live_trades SET qty = 0.4 WHERE ts::text LIKE '2026-05-07T01:03:31%'   AND symbol = 'SOL'   AND kind IN ('target_buy', 'target_sell', 'buy', 'sell')   AND ABS(qty - 0.3965299) < 0.01;  -- #12: 2026-05-07T01:03:31Z target_buy SOL 0.3965299 → 0.4
UPDATE live_trades SET qty = 9.0 WHERE ts::text LIKE '2026-05-07T01:03:31%'   AND symbol = 'XRP'   AND kind IN ('target_buy', 'target_sell', 'buy', 'sell')   AND ABS(qty - 8.93226977) < 0.01;  -- #13: 2026-05-07T01:03:31Z target_buy XRP 8.93226977 → 9.0
UPDATE live_trades SET qty = 0.42 WHERE ts::text LIKE '2026-05-06T13:03:33%'   AND symbol = 'SOL'   AND kind IN ('target_buy', 'target_sell', 'buy', 'sell')   AND ABS(qty - 0.41500174) < 0.01;  -- #14: 2026-05-06T13:03:33Z target_buy SOL 0.41500174 → 0.42
UPDATE live_trades SET qty = 261.0 WHERE ts::text LIKE '2026-05-05T20:00:42%'   AND symbol = 'SEI'   AND kind IN ('target_buy', 'target_sell', 'buy', 'sell')   AND ABS(qty - 260.73659939) < 0.01;  -- #15: 2026-05-05T20:00:42Z target_buy SEI 260.73659939 → 261.0
UPDATE live_trades SET qty = 1.5 WHERE ts::text LIKE '2026-05-05T20:00:41%'   AND symbol = 'LINK'   AND kind IN ('target_buy', 'target_sell', 'buy', 'sell')   AND ABS(qty - 1.51916293) < 0.01;  -- #16: 2026-05-05T20:00:41Z target_buy LINK 1.51916293 → 1.5
UPDATE live_trades SET qty = 2.6 WHERE ts::text LIKE '2026-05-05T19:00:43%'   AND symbol = 'LINK'   AND kind IN ('target_buy', 'target_sell', 'buy', 'sell')   AND ABS(qty - 2.6492713) < 0.01;  -- #17: 2026-05-05T19:00:43Z target_buy LINK 2.6492713 → 2.6
UPDATE live_trades SET qty = 3.2 WHERE ts::text LIKE '2026-05-05T19:00:42%'   AND symbol = 'INJ'   AND kind IN ('target_buy', 'target_sell', 'buy', 'sell')   AND ABS(qty - 3.17404151) < 0.01;  -- #18: 2026-05-05T19:00:42Z target_buy INJ 3.17404151 → 3.2

-- Verify no more mismatches:
-- SELECT ts, symbol, kind, qty FROM live_trades WHERE live_instance_id = 'mainnet-pilot' ORDER BY ts DESC LIMIT 50;

COMMIT;
