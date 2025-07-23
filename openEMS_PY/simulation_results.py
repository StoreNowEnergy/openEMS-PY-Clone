"""
simulation_results.py
=====================

Dataclass mirroring

  • io.openems.edge.energy.optimizer.SimulationResult

Holds the best schedule for a horizon **and** the aggregated KPIs that
OpenEMS’ UI later visualises.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass(slots=True)
class SimulationResult:
    # -------------- optimisation outcome --------------------------------
    best_schedule: Sequence[int]          # battery mode per Period index
    fitness: float                        # objective (lower = better)
    violated_constraints: int

    # -------------- KPIs (aggregated for the horizon) -------------------
    grid_buy_cost: float
    grid_sell_revenue: float

    @property
    def total_cost(self) -> float:
        return self.grid_buy_cost - self.grid_sell_revenue

    def as_numpy(self) -> np.ndarray:
        """Return the schedule as an int array (handy for GA/analytics)."""
        return np.asarray(self.best_schedule, dtype=np.int8)
