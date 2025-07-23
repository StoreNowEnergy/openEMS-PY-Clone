"""
initial_population_utils.py
===========================

Straight port of

    io.openems.edge.energy.optimizer.InitialPopulationUtils

Differences to the Java original
--------------------------------
* No `EshCodec` – we don’t need an integer <-> chromosome converter
  because DEAP works natively on `list[int]`.
* A *very* light‑weight `EshWithDifferentModes` protocol stands in for the
  Java interface.
* The result is returned as `List[List[int]]` (each inner list == entire
  chromosome), ready to feed into DEAP’s `Population`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List, Sequence, Protocol, Iterable, Optional
from collections import OrderedDict
import random

from global_optimization_context import GlobalOptimizationContext
from energy_schedule_handler import BatteryScheduleHandler


# --------------------------------------------------------------------------- 
#  Python shadow of InitialPopulation.Transition (just a thin dataclass)
# --------------------------------------------------------------------------- 
@dataclass(slots=True, frozen=True)
class Transition:
    """One component’s mode sequence for the whole horizon."""
    mode_indexes: Sequence[int]


# --------------------------------------------------------------------------- 
#  Minimal “interface” for components that offer different modes
#  (mirror of `EshWithDifferentModes`)
# --------------------------------------------------------------------------- 
class EshWithDifferentModes(Protocol):
    def parent_id(self) -> str: ...
    def default_mode_index(self) -> int: ...
    def get_initial_population(                   # Java: getInitialPopulation()
        self, goc: GlobalOptimizationContext
    ) -> Iterable[Transition]: ...


# --------------------------------------------------------------------------- 
#  Helper: create *all‑default* chromosome for one ESH
# --------------------------------------------------------------------------- 
def _all_default(
    goc: GlobalOptimizationContext, esh: EshWithDifferentModes
) -> Transition:
    return Transition([esh.default_mode_index()] * len(goc.periods))


# --------------------------------------------------------------------------- 
#  Helper: recreate schedule from *previous simulation result*
# --------------------------------------------------------------------------- 
def _from_previous_schedule(
    goc: GlobalOptimizationContext,
    esh: EshWithDifferentModes,
    previous_schedule: Optional[dict[int, list[int]]],  # {period_idx -> mode}
) -> Optional[Transition]:
    if previous_schedule is None:
        return None
    modes = [
        previous_schedule.get(p.index, esh.default_mode_index())
        for p in goc.periods
    ]
    return Transition(modes)


# --------------------------------------------------------------------------- 
#  Main builder (exact port of createInitialPopulationPerEsh)
# --------------------------------------------------------------------------- 
def _initial_population_for_esh(
    goc: GlobalOptimizationContext,
    esh: EshWithDifferentModes,
    previous_schedule: Optional[dict[int, list[int]]],
    is_first_period_fixed: bool,
) -> List[Transition]:
    """
    Return a *deduplicated* list (set in Java) of Transition seeds for
    ONE component/ESH.
    """
    seeds: "OrderedDict[tuple[int, ...], Transition]" = OrderedDict()

    def _add(t: Optional[Transition]):
        if t is None:
            return
        modes = list(t.mode_indexes)
        # ---- Java’s ‘applyIsCurrentPeriodFixed’ -------------------------
        if is_first_period_fixed and previous_schedule is not None:
            old_mode_0 = previous_schedule.get(0, esh.default_mode_index())
            modes[0] = old_mode_0
        seeds.setdefault(tuple(modes), Transition(modes))

    # 1. all‑default
    _add(_all_default(goc, esh))
    # 2. schedule from previous result
    _add(_from_previous_schedule(goc, esh, previous_schedule))
    # 3. handler‑provided seeds
    for t in esh.get_initial_population(goc):
        _add(t)

    return list(seeds.values())


# --------------------------------------------------------------------------- 
#  Public API: build whole GA initial population (cartesian product)
# --------------------------------------------------------------------------- 
def build_initial_population(
    goc: GlobalOptimizationContext,
    eshs: Sequence[EshWithDifferentModes],
    *,
    previous_result_schedules: Optional[
        dict[str, dict[int, list[int]]]
    ] = None,  # {eshId -> {periodIdx -> mode}}
    is_first_period_fixed: bool = True,
    ga_pop_size: int = 40,
    rng_seed: int = 42,
) -> List[List[int]]:
    """
    Parameters
    ----------
    goc
        Horizon for which we optimise.
    eshs
        All ‘components with different modes’ that must be encoded.
    previous_result_schedules
        Parsed copy of `SimulationResult.schedules()`; may be *None* on
        first run.
    is_first_period_fixed
        Mirror of the Java flag (keep current period constant).
    ga_pop_size
        Desired GA population.  If the cartesian product yields fewer
        individuals, we append random chromosomes until the size fits.
    """

    if previous_result_schedules is None:
        previous_result_schedules = {}

    # ---- 1. create seed lists per ESH -----------------------------------
    seed_lists_per_esh: List[List[Transition]] = []
    for esh in eshs:
        prev_sched = previous_result_schedules.get(esh.parent_id())
        seeds = _initial_population_for_esh(
            goc, esh, prev_sched, is_first_period_fixed
        )
        seed_lists_per_esh.append(seeds)

    # ---- 2. cartesian product into combined chromosomes -----------------
    population: List[List[int]] = []
    for combo in itertools.product(*seed_lists_per_esh):
        # combo is tuple[Transition,…]  –> merge component columns
        chromosome = []
        for period_idx, _ in enumerate(goc.periods):
            # append all ESHs’ mode for this period
            chromosome.extend(
                t.mode_indexes[period_idx] for t in combo
            )
        population.append(chromosome)

    # ---- 3. fill up with random individuals if necessary ----------------
    rng = random.Random(rng_seed)
    period_len = len(goc.periods) * len(eshs)
    # --- NEW: ask the handler for the highest valid mode value ----------
    handler = BatteryScheduleHandler()
    max_mode_value = handler.max_mode_value
    while len(population) < ga_pop_size:
        population.append(
            [rng.randint(0, max_mode_value) for _ in range(period_len)]
        )

    # ---- 4. trim if accidentally excessive ------------------------------
    return population[:ga_pop_size]
