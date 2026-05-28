#!/usr/bin/env python3
"""Fix historical trade qty in Postgres live_trades using Hyperliquid userFills API.

Read-only first pass: queries HL and the dashboard API, compares qty values,
outputs the mismatches and generates SQL UPDATE statements.

Usage:
    # Dry-run (just print mismatches, no SQL generated):
    python3 scripts/fix_historical_trade_qty.py --dry-run

    # Generate UPDATE SQL to stdout:
    python3 scripts/fix_historical_trade_qty.py --generate-sql > fix.sql

    # Apply to Postgres directly (requires DB access):
    python3 scripts/fix_historical_trade_qty.py --apply --db-url postgresql://...

Safety: only fixes trades where the dashboard qty differs from HL filled sz by
more than 0.1% (relative) or 0.001 (absolute), and only where the HL fill exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone

try:
    import urllib.request
except ImportError:
    pass  # type: ignore[import-not-found, no-redef]


HL_FILLS_URL = "https://api.hyperliquid.xyz/info"
HL_ADDRESS = "0x484807f90FbFc1e7578D4D6ABcaBad4990eBc678"
DASHBOARD_URL = "https://quant.rockydu.com/api/v1/live/cta-forge/public/latest"


def _http_get_json(url: str, data: bytes | None = None) -> dict | list:
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with (
        urllib.request.urlopen(  # nosemgrep: dynamic-urllib-use-detected — urls are hardcoded constants
            req, timeout=30
        ) as resp
    ):
        return json.loads(resp.read())


def fetch_hl_fills() -> list[dict]:
    """Fetch all user fills from Hyperliquid."""
    payload = json.dumps({"type": "userFills", "user": HL_ADDRESS}).encode()
    return list(_http_get_json(HL_FILLS_URL, data=payload))


def fetch_dashboard_events() -> list[dict]:
    """Fetch the current dashboard snapshot events."""
    raw = subprocess_run(["curl", "-s", "--max-time", "10", DASHBOARD_URL])
    snapshot = json.loads(raw)
    return [e for e in snapshot.get("events", []) if e.get("type") == "trade"]


def subprocess_run(cmd: list[str]) -> str:
    import subprocess

    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def build_hl_index(fills: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    """Index HL fills by (iso_minute, coin, side)."""
    index: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for f in fills:
        ts = f.get("time", 0)
        if not isinstance(ts, (int, float)):
            continue
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        key = (
            dt.strftime("%Y-%m-%dT%H:%M"),
            f.get("coin", "").strip(),
            f.get("side", ""),
        )
        index[key].append(f)
    return index


def match_trades(
    events: list[dict],
    hl_index: dict[tuple[str, str, str], list[dict]],
) -> list[dict]:
    """Match dashboard events to HL fills, identify qty mismatches.

    Returns list of dicts with keys:
        ts, symbol, dash_kind, dash_qty, hl_sz, hl_px, hl_side
    """
    mismatches = []
    for e in events:
        d_ts = e.get("time", "")
        if not d_ts:
            continue
        d_dt = datetime.fromisoformat(d_ts.replace("Z", "+00:00"))
        d_symbol = e.get("symbol", "").strip()
        d_kind = e.get("kind", "")
        d_qty = e.get("qty")

        # Map dashboard kind to HL side
        hl_side = "B" if "buy" in d_kind.lower() else "A"

        key = (d_dt.strftime("%Y-%m-%dT%H:%M"), d_symbol, hl_side)
        candidates = hl_index.get(key, [])

        if not candidates:
            continue

        total_sz = sum(float(h.get("sz", 0)) for h in candidates)
        hl_px = candidates[0].get("px", "0")

        if d_qty is None:
            continue

        # Only flag if difference is meaningful
        abs_diff = abs(d_qty - total_sz)
        if abs_diff < 0.0001:
            continue

        mismatches.append(
            {
                "ts": d_ts,
                "symbol": d_symbol,
                "dash_kind": d_kind,
                "dash_side": e.get("side", ""),
                "dash_qty": d_qty,
                "hl_sz": total_sz,
                "hl_px": float(hl_px),
                "hl_side": "buy" if hl_side == "B" else "sell",
            }
        )

    return mismatches


def generate_sql(mismatches: list[dict]) -> list[str]:
    """Generate UPDATE statements for Postgres live_trades table."""
    statements = []
    statements.append("-- Fix historical trade qty in live_trades")
    statements.append(f"-- Generated at {datetime.now(timezone.utc).isoformat()}")
    statements.append(f"-- {len(mismatches)} rows to update")
    statements.append("BEGIN;")
    statements.append("")

    for i, m in enumerate(mismatches, 1):
        # Match by ts + symbol + side (ts precision: full ISO timestamp)
        # Use the ts prefix since dashboard ts are stored with varying precision
        ts_prefix = m["ts"][:19]  # YYYY-MM-DDTHH:MM:SS
        sql = (
            f"UPDATE live_trades "
            f"SET qty = {m['hl_sz']} "
            f"WHERE ts::text LIKE '{ts_prefix}%' "
            f"  AND symbol = '{m['symbol']}' "
            f"  AND kind IN ('target_buy', 'target_sell', 'buy', 'sell') "
            f"  AND ABS(qty - {m['dash_qty']}) < 0.01;"
            f"  -- #{i}: {m['ts']} {m['dash_kind']} {m['symbol']} "
            f"{m['dash_qty']} → {m['hl_sz']}"
        )
        statements.append(sql)

    statements.append("")
    statements.append("-- Verify no more mismatches:")
    statements.append(
        "-- SELECT ts, symbol, kind, qty FROM live_trades "
        "WHERE live_instance_id = 'mainnet-pilot' ORDER BY ts DESC LIMIT 50;"
    )
    statements.append("")
    statements.append("COMMIT;")
    return statements


def apply_fixes(mismatches: list[dict], db_url: str) -> None:
    """Apply fixes directly to Postgres."""
    try:
        import psycopg2  # type: ignore[import-not-found]
    except ImportError:
        print(
            "Error: psycopg2 not installed. Install with: pip install psycopg2-binary",
            file=sys.stderr,
        )
        sys.exit(1)

    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    cur = conn.cursor()
    updated = 0

    try:
        for m in mismatches:
            ts_prefix = m["ts"][:19]
            cur.execute(
                "UPDATE live_trades SET qty = %s "
                "WHERE ts::text LIKE %s "
                "  AND symbol = %s "
                "  AND kind IN ('target_buy', 'target_sell', 'buy', 'sell') "
                "  AND ABS(qty - %s) < 0.01",
                (m["hl_sz"], f"{ts_prefix}%", m["symbol"], m["dash_qty"]),
            )
            if cur.rowcount and cur.rowcount > 0:
                updated += cur.rowcount
                print(
                    f"UPDATED: {m['ts']} {m['dash_kind']} {m['symbol']} "
                    f"{m['dash_qty']} → {m['hl_sz']}"
                )
            else:
                print(
                    f"SKIPPED (no match): {m['ts']} {m['dash_kind']} {m['symbol']}",
                    file=sys.stderr,
                )

        conn.commit()
        print(f"\n{updated} rows updated successfully.")
    except Exception as exc:
        conn.rollback()
        print(f"Error applying fixes: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix historical trade qty using HL userFills"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print mismatches without generating SQL (default)",
    )
    parser.add_argument(
        "--generate-sql",
        action="store_true",
        help="Generate UPDATE SQL statements to stdout",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply fixes directly to Postgres",
    )
    parser.add_argument(
        "--db-url",
        default="",
        help="Postgres connection URL (required for --apply)",
    )
    args = parser.parse_args()

    print("Fetching HL fills...", file=sys.stderr)
    hl_fills = fetch_hl_fills()
    print(f"  Got {len(hl_fills)} fills", file=sys.stderr)

    print("Fetching dashboard events...", file=sys.stderr)
    dash_events = fetch_dashboard_events()
    print(f"  Got {len(dash_events)} trade events", file=sys.stderr)

    print("Matching...", file=sys.stderr)
    hl_index = build_hl_index(hl_fills)
    mismatches = match_trades(dash_events, hl_index)

    if not mismatches:
        print("No mismatches found. All trades match.", file=sys.stderr)
        return

    print(f"Found {len(mismatches)} mismatches:", file=sys.stderr)
    print()

    for m in mismatches:
        print(
            f"  {m['ts']} {m['dash_kind']:>12} {m['symbol']:>5} "
            f"dash={m['dash_qty']} → hl={m['hl_sz']} "
            f"(diff={m['hl_sz'] - m['dash_qty']:+.6f})"
        )

    if args.generate_sql:
        print()
        sql_lines = generate_sql(mismatches)
        for line in sql_lines:
            print(line)

    if args.apply:
        if not args.db_url:
            print("Error: --db-url is required for --apply", file=sys.stderr)
            sys.exit(1)
        print()
        apply_fixes(mismatches, args.db_url)


if __name__ == "__main__":
    main()
