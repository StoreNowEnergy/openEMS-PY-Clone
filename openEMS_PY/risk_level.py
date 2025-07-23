"""
risk_level.py
=============

Python port of `io.openems.edge.energy.api.RiskLevel`.

Only the three RiskLevels that currently exist in the Java enum are exposed.
Each enum member’s *value* is the corresponding `efficiencyFactor`.

    • LOW     → 1.20
    • MEDIUM  → 1.17   (OpenEMS default)
    • HIGH    → 1.10
"""

from enum import Enum


class RiskLevel(Enum):
    LOW = 1.20
    MEDIUM = 1.17
    HIGH = 1.10

    # --------------------------------------------------------------------- #
    # Convenience helpers (match Java API shape)
    # --------------------------------------------------------------------- #

    @property
    def efficiency_factor(self) -> float:     # same public name as Java
        """Return the numeric efficiencyFactor used in cost calculation."""
        return self.value

    # allow `.factor` as a shorter alias
    factor = efficiency_factor

