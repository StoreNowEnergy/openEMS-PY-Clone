"""
System parameters used by the simulation.

This module defines a single dataclass that collects the electrical
storage system (ESS) and grid constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

from openEMS_PY.global_optimization_context import Ess, Grid


@dataclass(slots=True)
class SystemParameters:
    """Container for ESS and grid parameters."""

    # ESS parameters
    capacity_kwh: float = 5.0
    max_charge_kw: float = 3.68
    max_discharge_kw: float = 3.68
    charge_eff: float = 0.95
    discharge_eff: float = 0.95
    min_soc_pct: float = 0.15
    max_soc_pct: float = 0.90

    # Grid parameters
    grid_max_buy_kw: float = 10.0
    grid_max_sell_kw: float = 6.0

    # ---------------------------------------------------------------
    # Convenience helpers
    # ---------------------------------------------------------------
    @property
    def min_soc_kwh(self) -> float:
        return self.capacity_kwh * self.min_soc_pct

    @property
    def max_soc_kwh(self) -> float:
        return self.capacity_kwh * self.max_soc_pct

    def as_ess(self) -> Ess:
        """Return an :class:`Ess` instance with these parameters."""
        return Ess(
            capacity_kwh=self.capacity_kwh,
            max_charge_kw=self.max_charge_kw,
            max_discharge_kw=self.max_discharge_kw,
            charge_eff=self.charge_eff,
            discharge_eff=self.discharge_eff,
            min_soc_pct=self.min_soc_pct,
            max_soc_pct=self.max_soc_pct,
        )

    def as_grid(self) -> Grid:
        """Return a :class:`Grid` instance with these parameters."""
        return Grid(
            max_buy_kw=self.grid_max_buy_kw,
            max_sell_kw=self.grid_max_sell_kw,
        )
