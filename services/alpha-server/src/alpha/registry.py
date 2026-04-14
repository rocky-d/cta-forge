"""Alpha factor registry — auto-discovery and hot-plug."""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cta_core.protocols import AlphaFactor

from . import factors as _factors_pkg

logger = logging.getLogger(__name__)


class FactorRegistry:
    """Registry for alpha factors. Discovers factors from the factors/ package."""

    def __init__(self) -> None:
        self._factors: dict[str, AlphaFactor] = {}

    def register(self, factor: AlphaFactor) -> None:
        self._factors[factor.name] = factor
        logger.info("Registered factor: %s", factor.name)

    def get(self, name: str) -> AlphaFactor | None:
        return self._factors.get(name)

    def list_factors(self) -> list[str]:
        return sorted(self._factors.keys())

    def all(self) -> dict[str, AlphaFactor]:
        return dict(self._factors)

    def auto_discover(self) -> None:
        """Scan the factors/ package and register all classes with a `name` property and `compute` method."""
        for _importer, modname, _ispkg in pkgutil.iter_modules(_factors_pkg.__path__):
            module = importlib.import_module(
                f".factors.{modname}", package="alpha"
            )
            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                if (
                    isinstance(obj, type)
                    and hasattr(obj, "name")
                    and hasattr(obj, "compute")
                    and attr_name != "AlphaFactor"
                ):
                    try:
                        instance = obj()
                        self.register(instance)
                    except Exception:
                        logger.exception(
                            "Failed to instantiate factor %s.%s", modname, attr_name
                        )


# Global registry
registry = FactorRegistry()
