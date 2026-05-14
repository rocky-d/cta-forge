"""Guardrails for the schema-only live persistence migration."""

from pathlib import Path

SCHEMA = Path("infra/db/migrations/001_live_persistence.sql")


def _schema_sql() -> str:
    return SCHEMA.read_text()


def _table_block(sql: str, table_name: str) -> str:
    marker = f"create table if not exists {table_name} ("
    start = sql.index(marker)
    end = sql.index("\n);", start)
    return sql[start:end]


def test_live_persistence_schema_covers_core_tables() -> None:
    sql = _schema_sql()

    for table_name in [
        "strategies",
        "strategy_profiles",
        "exchange_accounts",
        "live_instances",
        "live_runs",
        "engine_checkpoints",
        "live_ticks",
        "live_positions",
        "live_targets",
        "live_trades",
        "live_signals",
        "public_dashboard_instances",
    ]:
        assert f"create table if not exists {table_name}" in sql


def test_live_persistence_schema_preserves_decimal_precision() -> None:
    sql = _schema_sql().lower()

    assert "double precision" not in sql
    assert " real" not in sql
    assert "account_equity numeric not null" in sql
    assert "target_gross numeric not null" in sql
    assert "qty numeric not null" in sql


def test_public_dashboard_instance_table_stays_public_safe() -> None:
    block = _table_block(_schema_sql(), "public_dashboard_instances").lower()

    for private_column in [
        "account_id",
        "address",
        "wallet",
        "balance",
        "order_id",
        "raw_json",
        "run_id",
    ]:
        assert private_column not in block


def test_public_instance_slug_required_before_public_enablement() -> None:
    block = _table_block(_schema_sql(), "live_instances").lower()

    assert "check (not public_enabled or public_instance_slug is not null)" in block


def test_position_rows_cannot_reference_a_different_instance_than_tick() -> None:
    block = _table_block(_schema_sql(), "live_positions").lower()

    assert "foreign key (tick_id, live_instance_id)" in block
    assert "references live_ticks(id, live_instance_id) on delete cascade" in block
