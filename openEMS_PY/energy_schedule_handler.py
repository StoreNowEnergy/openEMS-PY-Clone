"""
energy_schedule_handler.py
==========================

Very small subset of the huge Java interface – only what the optimiser
needs right now:

* a *parent‑id* (string) so results can be stored per component
* a list of **different battery modes** (0… n‑1)
* helper for the optimiser to ask…
      • get_default_mode_index()
      • get_initial_population()
      • max_mode_value           (for random chromosomes)
"""

from __future__ import annotations
from typing import Iterable, Sequence
from dataclasses import dataclass

from global_optimization_context import GlobalOptimizationContext
from initial_population_utils import Transition


# --------------------------------------------------------------------- #
#  Battery with 5 standard modes (same numbers as used in simulator.py) #
# --------------------------------------------------------------------- #
@dataclass(slots=True)
class BatteryScheduleHandler:
    modes: Sequence[str] = ("CHARGE", "DISCHARGE", "IDLE", "AUTO_PRICE", "AUTO_PV")
    @property
    def max_mode_value(self):
        return len(self.modes) - 1


    # ------------------------------------------------------------------ #
    #  API expected by initial_population_utils                          #
    # ------------------------------------------------------------------ #
    def parent_id_(self) -> str:           # different name to avoid clash
        return self.parent_id

    def default_mode_index(self) -> int:
        return 2                           # 'IDLE'

    def get_initial_population(
        self, goc: GlobalOptimizationContext
    ) -> Iterable[Transition]:
        """
        Provide *component specific* seed schedules.
        Here:  ➜ always AUTO_PV  (simple demo).
        """
        n = len(goc.periods)
        yield Transition([4] * n)          # all AUTO_PV

    # helper for random individuals
    @property
    def max_mode_value(self) -> int:
        return len(self.MODES) - 1
