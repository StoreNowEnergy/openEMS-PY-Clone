"""
simulator.py
============

Re‑implementation of

  • io.openems.edge.energy.optimizer.Simulator

Responsibilities
----------------
* Generate an initial population (ported from InitialPopulationUtils.java)
* Evolve it with a tiny GA (DEAP, 3–5 generations – good enough for a demo)
* For each candidate schedule: call **energy_flow.solve_energy_flow** period
  by period and accumulate Fitness – identical to Java.

❗ Nothing else – no plotting, KPI printing, etc.  Those remain in your
analysis script just like OpenEMS keeps KPI/UI code outside the optimizer.
"""

from __future__ import annotations

import random
from typing import List

import numpy as np
from deap import base, creator, tools

from .global_optimization_context import GlobalOptimizationContext
from .energy_flow import solve_energy_flow, EnergyFlowResult
from .simulation_results import SimulationResult


# ---------------------------------------------------------------------------#
#  GA parameters (keep tiny – Python is slower than Java)                    #
# ---------------------------------------------------------------------------#
POP_SIZE = 14
N_GENERATIONS = 4
CXPB, MUTPB = 0.7, 0.3


# ---------------------------------------------------------------------------#
#  Fitness helper (OpenEMS three‑level comparison)                           #
# ---------------------------------------------------------------------------#
class FitnessAccumulator:
    __slots__ = ("violations", "grid_buy_cost", "grid_sell_revenue")

    def __init__(self) -> None:
        self.violations = 0
        self.grid_buy_cost = 0.0
        self.grid_sell_revenue = 0.0

    def add(self, flow: EnergyFlowResult) -> None:
        self.violations += flow.violated_constraints
        self.grid_buy_cost += flow.grid_buy_cost
        self.grid_sell_revenue += flow.grid_sell_revenue

    # -------- numeric fitness used by GA --------------------------------
    def value(self) -> float:
        """
        Reproduce the lexicographical comparison from EnergyScheduleHandler:
            1) fewer constraint violations
            2) lower buy cost
            3) higher sell revenue
        We merge them into a single scalar by huge weights.
        """
        return (
            self.violations * 1e9      # dominate everything
            + self.grid_buy_cost
            - self.grid_sell_revenue
        )


# ---------------------------------------------------------------------------#
#  Initial population (= InitialPopulationUtils.java)                        #
# ---------------------------------------------------------------------------#
def build_initial_population(ctx: GlobalOptimizationContext) -> List[List[int]]:
    """
    Six heuristics copied 1‑to‑1 from InitialPopulationUtils.
    Modes:
        0 CHARGE, 1 DISCHARGE, 2 IDLE, 3 AUTO_PRICE, 4 AUTO_PV
    """
    Q, H = 24, 18
    n = len(ctx.periods)
    assert n == Q + H, "expect default 24h horizon"

    CHARGE, DISCHARGE, IDLE, AUTO_PRICE, AUTO_PV = range(5)

    pop: List[List[int]] = []

    # 1) all idle
    pop.append([IDLE] * n)

    # 2) cheap‑charge expensive‑discharge
    prices = np.array([p.price_buy for p in ctx.periods])
    cheap_idx = prices.argsort()[:6]
    expensive_idx = prices.argsort()[-6:]
    schedule = [IDLE] * n
    for i in cheap_idx:
        schedule[i] = CHARGE
    for i in expensive_idx:
        schedule[i] = DISCHARGE
    pop.append(schedule)

    # 3) PV self‑consumption
    schedule = [IDLE] * n
    for i, p in enumerate(ctx.periods):
        if p.production_kwh > p.consumption_kwh:
            schedule[i] = CHARGE
    pop.append(schedule)

    # 4) day/night heuristic
    schedule = [IDLE] * n
    for i, p in enumerate(ctx.periods):
        hour = p.time.hour
        schedule[i] = CHARGE if 9 <= hour < 16 else DISCHARGE if 18 <= hour < 23 else IDLE
    pop.append(schedule)

    # 5) all AUTO_PV
    pop.append([AUTO_PV] * n)

    # 6) random for diversity
    pop.append([random.randint(0, 4) for _ in range(n)])

    return pop


# ---------------------------------------------------------------------------#
#  GA toolbox setup                                                          #
# ---------------------------------------------------------------------------#
def _setup_deap(n_periods: int):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_mode", random.randint, 0, 4)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_mode, n=n_periods)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=4, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox


# ---------------------------------------------------------------------------#
#  Main entry point                                                          #
# ---------------------------------------------------------------------------#
def simulate(ctx: GlobalOptimizationContext) -> SimulationResult:
    """
    Optimise one 24 h horizon and return the best schedule + KPIs.
    """

    n_p = len(ctx.periods)
    toolbox = _setup_deap(n_p)

    # ----- seed initial pop with heuristics ------------------------------
    population = [
        creator.Individual(h)                # heuristics
        for h in build_initial_population(ctx)
    ]
    population.extend(toolbox.population(n=POP_SIZE - len(population)))

    # ------------------------------------------------------------------ #
    # Evaluation function  (wrapped to close over ctx)                   #
    # ------------------------------------------------------------------ #
    def eval_schedule(individual: List[int]):
        soc = ctx.initial_soc_kwh
        fit = FitnessAccumulator()
        for mode_int, period in zip(individual, ctx.periods):
            flow = solve_energy_flow(
                ctx, period, soc_kwh=soc, risk_factor=ctx.risk_factor
            )
            fit.add(flow)
            # update SoC (same mathematics as Java Simulator)
            if flow.ess_net > 0:
                soc -= flow.ess_net / ctx.ess.discharge_eff
            else:
                soc -= flow.ess_net * ctx.ess.charge_eff
            soc = np.clip(soc, ctx.ess.min_soc_kwh, ctx.ess.max_soc_kwh)
        return (fit.value(),)

    toolbox.register("evaluate", eval_schedule)

    # ----- GA loop -------------------------------------------------------
    invalid = [ind for ind in population if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid))
    for ind, f in zip(invalid, fitnesses):
        ind.fitness.values = f

    for _ in range(N_GENERATIONS):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # crossover + mutation
        for i in range(1, len(offspring), 2):
            if random.random() < CXPB:
                toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values, offspring[i].fitness.values
        for ind in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(ind)
                del ind.fitness.values

        # re‑evaluate
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid))
        for ind, f in zip(invalid, fitnesses):
            ind.fitness.values = f

        population[:] = offspring

    # ----- best individual ----------------------------------------------
    best = tools.selBest(population, k=1)[0]
    fitness_value = best.fitness.values[0]

    # Re‑simulate *once* to obtain detailed KPIs
    soc = ctx.initial_soc_kwh
    kpi = FitnessAccumulator()
    violations = 0
    ess_net_sum = 0.0
    for mode_int, period in zip(best, ctx.periods):
        flow = solve_energy_flow(ctx, period, soc, ctx.risk_factor)
        kpi.add(flow)
        violations += flow.violated_constraints
        ess_net_sum += flow.ess_net
        soc += 0  # SoC update not needed for KPIs beyond 24 h window

    return SimulationResult(
        best_schedule=list(best),
        fitness=fitness_value,
        violated_constraints=violations,
        grid_buy_cost=kpi.grid_buy_cost,
        grid_sell_revenue=kpi.grid_sell_revenue,
        ess_net_kwh=ess_net_sum,
        time=ctx.periods[0].time,
    )

