-- cta-forge live persistence schema, v1.
--
-- Scope: live runtime identity, checkpoints, journal-equivalent events, and
-- public dashboard instance metadata. This intentionally does not migrate
-- parquet market data yet.

begin;

create table if not exists strategies (
    slug text primary key,
    name text not null,
    created_at timestamptz not null default now()
);

create table if not exists strategy_profiles (
    profile_id text primary key,
    strategy_slug text not null references strategies(slug),
    slug text not null,
    version text not null default '',
    config_json jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (strategy_slug, slug, version)
);

create table if not exists exchange_accounts (
    account_id text primary key,
    exchange text not null,
    network text not null check (network in ('testnet', 'mainnet')),
    account_label text not null,
    address_hash text,
    address_prefix text,
    status text not null default 'active'
        check (status in ('active', 'paused', 'retired')),
    created_at timestamptz not null default now()
);

create table if not exists live_instances (
    live_instance_id text primary key,
    strategy_slug text not null references strategies(slug),
    profile_id text not null references strategy_profiles(profile_id),
    account_id text not null references exchange_accounts(account_id),
    public_instance_slug text,
    mode text not null
        check (mode in ('dry_run', 'testnet_live', 'mainnet_pilot', 'mainnet_live')),
    status text not null default 'active'
        check (status in ('active', 'paused', 'retired')),
    risk_config_json jsonb not null default '{}'::jsonb,
    public_enabled boolean not null default false,
    created_at timestamptz not null default now(),
    check (not public_enabled or public_instance_slug is not null),
    unique (strategy_slug, public_instance_slug)
);

create table if not exists live_runs (
    run_id text primary key,
    live_instance_id text not null references live_instances(live_instance_id),
    git_sha text,
    image_ref text,
    started_at timestamptz not null default now(),
    stopped_at timestamptz,
    runtime_config_json jsonb not null default '{}'::jsonb,
    status text not null default 'running'
        check (status in ('running', 'stopped', 'failed'))
);

create table if not exists engine_checkpoints (
    live_instance_id text primary key references live_instances(live_instance_id),
    run_id text not null references live_runs(run_id),
    bar_count integer not null check (bar_count >= 0),
    payload_json jsonb not null,
    saved_at timestamptz not null default now()
);

create table if not exists live_ticks (
    id bigint generated always as identity primary key,
    live_instance_id text not null references live_instances(live_instance_id),
    run_id text not null references live_runs(run_id),
    bar integer not null check (bar >= 0),
    ts timestamptz not null,
    account_equity numeric not null,
    peak_equity numeric not null,
    dd_pct numeric not null,
    n_positions integer not null check (n_positions >= 0),
    status text not null default 'ok',
    summary_json jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (live_instance_id, bar),
    unique (id, live_instance_id)
);

create table if not exists live_positions (
    tick_id bigint not null,
    live_instance_id text not null,
    symbol text not null,
    side text not null check (side in ('long', 'short')),
    qty numeric not null,
    entry_price numeric,
    best_price numeric,
    raw_json jsonb not null default '{}'::jsonb,
    primary key (tick_id, symbol),
    foreign key (tick_id, live_instance_id)
        references live_ticks(id, live_instance_id) on delete cascade
);

create table if not exists live_targets (
    id bigint generated always as identity primary key,
    live_instance_id text not null references live_instances(live_instance_id),
    run_id text not null references live_runs(run_id),
    bar integer not null check (bar >= 0),
    ts timestamptz not null,
    profile text not null,
    target_ts timestamptz,
    staleness_seconds numeric,
    target_gross numeric not null,
    normalized_gross numeric not null,
    ignored_gross numeric not null default 0,
    ignored_gross_ratio numeric not null default 0,
    execution_coverage numeric not null,
    weights_json jsonb not null default '{}'::jsonb,
    ignored_weights_json jsonb not null default '{}'::jsonb,
    orders_json jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now(),
    unique (live_instance_id, bar, profile, target_ts)
);

create table if not exists live_trades (
    id bigint generated always as identity primary key,
    live_instance_id text not null references live_instances(live_instance_id),
    run_id text not null references live_runs(run_id),
    bar integer not null check (bar >= 0),
    ts timestamptz not null,
    kind text not null,
    symbol text not null,
    side text,
    qty numeric not null,
    price numeric not null,
    reason text not null,
    pnl numeric,
    pnl_pct numeric,
    held_bars integer,
    exchange_order_id text,
    fee numeric,
    raw_json jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (live_instance_id, run_id, bar, ts, kind, symbol, qty, price, reason)
);

create table if not exists live_signals (
    id bigint generated always as identity primary key,
    live_instance_id text not null references live_instances(live_instance_id),
    run_id text not null references live_runs(run_id),
    bar integer not null check (bar >= 0),
    ts timestamptz not null,
    signals_json jsonb not null,
    created_at timestamptz not null default now(),
    unique (live_instance_id, bar)
);

create table if not exists public_dashboard_instances (
    strategy_slug text not null references strategies(slug),
    public_instance_slug text not null,
    live_instance_id text not null references live_instances(live_instance_id),
    display_name text not null,
    status text not null default 'hidden'
        check (status in ('live', 'stale', 'paused', 'retired', 'hidden')),
    is_default boolean not null default false,
    sort_order integer not null default 100,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    primary key (strategy_slug, public_instance_slug),
    unique (strategy_slug, live_instance_id)
);

create unique index if not exists public_dashboard_instances_one_default
    on public_dashboard_instances(strategy_slug)
    where is_default;

create index if not exists live_runs_instance_started_idx
    on live_runs(live_instance_id, started_at desc);

create index if not exists live_ticks_instance_ts_idx
    on live_ticks(live_instance_id, ts desc);

create index if not exists live_targets_instance_ts_idx
    on live_targets(live_instance_id, ts desc);

create index if not exists live_trades_instance_ts_idx
    on live_trades(live_instance_id, ts desc);

create index if not exists live_signals_instance_ts_idx
    on live_signals(live_instance_id, ts desc);

commit;
