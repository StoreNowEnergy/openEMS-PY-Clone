"""
simulation_results.py
=====================

Dataclass mirroring

  • io.openems.edge.energy.optimizer.SimulationResult

The Python port returns the KPIs for **the first 15 minute period** of the
optimisation horizon.  This allows a rolling simulation where each call only
executes the earliest step and advances the state of charge accordingly.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Sequence, Optional, Dict, Any
import numpy as np


@dataclass(slots=True)
class SimulationResult:
    """Outcome of one optimisation horizon."""

    # -------------- optimisation outcome --------------------------------
    best_schedule: Sequence[int]          # battery mode per Period index
    fitness: float                        # objective (lower = better)
    violated_constraints: int

    # -------------- KPIs for the FIRST period ---------------------------
    grid_buy_cost: float
    grid_sell_revenue: float
    ess_net_kwh: float                     # positive == discharge
    grid_to_ess: float                    # +charge from grid, −ESS→grid
    prod_to_ess: float                    # PV charged into ESS
    ess_to_cons: float                    # discharge to consumption
    prod_to_grid: float                   # PV sold to grid
    time: Optional[datetime] = None        # start time of horizon

    @property
    def total_cost(self) -> float:
        return self.grid_buy_cost - self.grid_sell_revenue

    def as_numpy(self) -> np.ndarray:
        """Return the schedule as an int array (handy for GA/analytics)."""
        return np.asarray(self.best_schedule, dtype=np.int8)

    def as_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of all dataclass fields."""
        return asdict(self)
