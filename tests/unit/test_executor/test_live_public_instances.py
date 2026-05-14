from executor.live_public_instances import (
    PublicDashboardInstance,
    build_public_instances_payload,
)


def test_build_public_instances_payload_excludes_private_identity() -> None:
    payload = build_public_instances_payload(
        strategy_slug="cta-forge",
        instances=[
            PublicDashboardInstance(
                public_instance_slug="mainnet-pilot",
                display_name="Mainnet Pilot",
                status="active",
                is_default=True,
            )
        ],
    )

    assert payload == {
        "schema_version": "dashboard.public_instances.v1",
        "strategy_slug": "cta-forge",
        "default_instance_slug": "mainnet-pilot",
        "instances": [
            {
                "public_instance_slug": "mainnet-pilot",
                "display_name": "Mainnet Pilot",
                "status": "active",
                "is_default": True,
            }
        ],
    }
    assert "live_instance_id" not in str(payload)
    assert "account_id" not in str(payload)


def test_build_public_instances_payload_uses_first_instance_when_no_default() -> None:
    payload = build_public_instances_payload(
        strategy_slug="cta-forge",
        instances=[
            PublicDashboardInstance("pilot-a", "Pilot A", "active"),
            PublicDashboardInstance("pilot-b", "Pilot B", "active"),
        ],
    )

    assert payload["default_instance_slug"] == "pilot-a"


def test_build_public_instances_payload_handles_empty_list() -> None:
    payload = build_public_instances_payload(strategy_slug="cta-forge", instances=[])

    assert payload["default_instance_slug"] is None
    assert payload["instances"] == []
