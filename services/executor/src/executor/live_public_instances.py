"""Public-safe live dashboard instance projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class PublicDashboardInstance:
    """Public metadata for one dashboard-selectable live instance."""

    public_instance_slug: str
    display_name: str
    status: str
    is_default: bool = False


def build_public_instances_payload(
    *,
    strategy_slug: str,
    generated_at: str,
    instances: Sequence[PublicDashboardInstance],
) -> dict[str, Any]:
    """Build the public instance-list payload without private identifiers."""

    default_instance_slug = _default_instance_slug(instances)
    return {
        "schema_version": "dashboard.public_instances.v1",
        "generated_at": generated_at,
        "strategy": {"slug": strategy_slug},
        "default_instance_slug": default_instance_slug,
        "instances": [
            {
                "public_instance_slug": instance.public_instance_slug,
                "display_name": instance.display_name,
                "status": instance.status,
                "is_default": instance.is_default,
            }
            for instance in instances
        ],
    }


def _default_instance_slug(
    instances: Sequence[PublicDashboardInstance],
) -> str | None:
    if not instances:
        return None
    default = next((instance for instance in instances if instance.is_default), None)
    return (default or instances[0]).public_instance_slug
