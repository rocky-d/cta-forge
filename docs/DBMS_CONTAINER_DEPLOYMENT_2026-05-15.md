# DBMS container deployment plan - 2026-05-15

Purpose: run cta-forge live persistence PostgreSQL as part of the existing EC2 Docker Compose deployment, without introducing a separate host-managed database service or a heavier secret manager yet.

## Current decision

- PostgreSQL runs as a `postgres:17-alpine` Compose service named `postgres`.
- Data persists in the named Docker volume `cta_forge_postgres_data`.
- The executor image owns the checked-in schema migration SQL and exposes a small migration CLI.
- GitHub Actions deploy keeps building/pushing only the executor image, then on EC2:
  1. pulls images;
  2. starts/keeps the `postgres` container up;
  3. runs the one-shot `db-migrate` service;
  4. runs strict mainnet preflight;
  5. recreates only `executor-live`.
- The deployment `.env` remains the practical private env source for now.
- No PostgreSQL port is published to the internet.

## Why this shape

- Minimal operational moving parts: one Compose project on one EC2 host.
- Fits the current manual GitHub Actions deploy flow.
- Keeps DB lifecycle visible in deploy logs without printing secrets.
- Avoids host package PostgreSQL as a hidden dependency.
- Keeps future migration path open for RDS/SSM/Secrets Manager later.

## Runtime safety

- `PERSISTENCE_BACKEND=file` remains the default and current live source of truth.
- Having `DATABASE_URL` configured does not enable DB writes by itself.
- Dual-write requires a separate explicit runtime config change and observation plan.
- PostgreSQL migration is idempotent because schema uses `create table if not exists` / `create index if not exists`.

## Private EC2 env additions

The EC2 private `.env` should include:

```dotenv
POSTGRES_DB=cta_forge_live
POSTGRES_USER=cta_forge_live
POSTGRES_PASSWORD=<private generated value>
DATABASE_URL=postgresql://cta_forge_live:<same private value>@postgres:5432/cta_forge_live
LIVE_INSTANCE_ID=mainnet-pilot
PERSISTENCE_BACKEND=file
LIVE_PERSISTENCE_SHADOW_FAILURE_POLICY=warn
```

Do not commit these values. Do not print `DATABASE_URL` in logs.

## Deployment boundary

This change prepares the database container and migrations. It does not import historical records, enable dual-write, or make PostgreSQL source of truth. Those remain separate gates.
