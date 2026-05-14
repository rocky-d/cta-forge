# cta-forge DB migrations

This directory holds SQL migrations for the planned PostgreSQL-backed live
persistence layer.

Current scope:

- live runtime identity;
- live run/checkpoint records;
- journal-equivalent tick, target, trade, signal, and position tables;
- public dashboard instance metadata for future multi-live switching.

Out of scope for the first migration:

- private key or secret storage;
- parquet market data migration;
- runtime DB wiring;
- dashboard API changes.

The first rollout should import historical JSONL/state into PostgreSQL and
validate DB-derived reports against the existing file-derived reports before any
live read-path cutover.
