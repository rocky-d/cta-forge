alter table live_targets
    add column if not exists planned_orders_json jsonb not null default '[]'::jsonb,
    add column if not exists submitted_orders_json jsonb not null default '[]'::jsonb,
    add column if not exists filled_trades_json jsonb not null default '[]'::jsonb,
    add column if not exists failed_orders_json jsonb not null default '[]'::jsonb;

update live_targets
set planned_orders_json = orders_json
where planned_orders_json = '[]'::jsonb
  and orders_json <> '[]'::jsonb;
