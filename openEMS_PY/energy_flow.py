"""
energy_flow.py
==============

Python re‑implementation of

  • io.openems.edge.energy.api.simulation.EnergyFlow

It creates a *single* linear program per timestep and solves it with
SciPy’s `linprog(method="highs")`.  The return value is an
`EnergyFlowResult` dataclass that mirrors the Java record.

No other package dependencies except NumPy + SciPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple

import numpy as np
from scipy.optimize import linprog

from .global_optimization_context import Period, GlobalOptimizationContext


# ---------------------------------------------------------------------------#
#  Variable indices (mirror Java enum EnergyFlowCoefficient)                 #
# ---------------------------------------------------------------------------#

class C(IntEnum):
    PROD             = 0   # generation (PV)
    CONS             = 1   # consumption (load)
    ESS              = 2   # battery net (+ discharge, − charge)
    GRID             = 3   # grid net   (+ export,    − import)
    PROD_TO_CONS     = 4
    PROD_TO_ESS      = 5
    PROD_TO_GRID     = 6
    GRID_TO_CONS     = 7
    GRID_TO_ESS      = 8
    ESS_TO_CONS      = 9   # — 10 variables total


# ---------------------------------------------------------------------------#
#  Result container (like Java EnergyFlow record)                             #
# ---------------------------------------------------------------------------#

@dataclass(slots=True)
class EnergyFlowResult:
    """
    All flows [kWh] **per period** and the derived cost / revenue.
    """

    # --- decision variables -------------------------------------------
    prod_to_ess: float
    prod_to_cons: float
    prod_to_grid: float
    ess_to_cons: float
    grid_to_ess: float
    grid_to_cons: float
    ess_net: float
    grid_net: float

    # --- KPIs ----------------------------------------------------------
    grid_buy_cost: float
    grid_sell_revenue: float
    violated_constraints: int


# ---------------------------------------------------------------------------#
#  Core solver                                                               #
# ---------------------------------------------------------------------------#

def solve_energy_flow(
    goc: GlobalOptimizationContext,
    period: Period,
    soc_kwh: float,
    risk_factor: float,
) -> EnergyFlowResult:
    """
    Build the LP shown in EnergyFlow.java §“buildLinearProgram()”,
    solve it and pack the outcome in EnergyFlowResult.
    """

    # ------------------------------------------------------------------ #
    # Convenience aliases                                                #
    # ------------------------------------------------------------------ #
    ess = goc.ess
    grid = goc.grid
    hrs = period.duration.hours

    # maximum (dis)charge energy this period
    max_charge_kwh = ess.max_charge_kw * hrs
    max_discharge_kwh = ess.max_discharge_kw * hrs

    # SoC‑based limits
    charge_room   = max(0.0, ess.max_soc_kwh - soc_kwh) / ess.charge_eff
    discharge_room = max(0.0, soc_kwh - ess.min_soc_kwh) * ess.discharge_eff

    charge_cap   = min(max_charge_kwh, charge_room)
    discharge_cap = min(max_discharge_kwh, discharge_room)

    # ------------------------------------------------------------------ #
    # Objective   (minimise cost)                                        #
    # ------------------------------------------------------------------ #
    c = np.zeros(len(C))
    c[C.GRID_TO_CONS] = period.price_buy
    c[C.GRID_TO_ESS]  = period.price_buy * risk_factor     # <‑‑ risk factor
    c[C.PROD_TO_GRID] = -period.price_sell                 # revenue negative

    # ------------------------------------------------------------------ #
    # Equality constraints  A_eq · x  = b_eq                            #
    # (order identical to Java code – see comments there)               #
    # ------------------------------------------------------------------ #
    A_eq, b_eq = [], []

    # 1) PROD + GRID + ESS = CONS
    row = np.zeros(len(C))
    row[[C.PROD, C.GRID, C.ESS]] = 1
    row[C.CONS] = -1
    A_eq.append(row)
    b_eq.append(0)

    # 2) PROD distribution
    row = np.zeros(len(C))
    row[C.PROD]             = -1
    row[[C.PROD_TO_CONS,
         C.PROD_TO_ESS,
         C.PROD_TO_GRID]]   = 1
    A_eq.append(row)
    b_eq.append(0)

    # 3) CONS distribution
    row = np.zeros(len(C))
    row[C.CONS]             = 1
    row[[C.PROD_TO_CONS,
         C.GRID_TO_CONS,
         C.ESS_TO_CONS]]    = -1
    A_eq.append(row)
    b_eq.append(0)

    # 4) GRID distribution
    row = np.zeros(len(C))
    row[C.GRID]             = -1
    row[[C.GRID_TO_CONS,
         C.GRID_TO_ESS]]    = 1
    row[C.PROD_TO_GRID]     = -1
    A_eq.append(row)
    b_eq.append(0)

    # 5) ESS distribution
    row = np.zeros(len(C))
    row[C.ESS]              = -1
    row[C.ESS_TO_CONS]      = 1
    row[[C.PROD_TO_ESS,
         C.GRID_TO_ESS]]    = -1
    A_eq.append(row)
    b_eq.append(0)

    # 6) fixed production
    row = np.zeros(len(C)); row[C.PROD] = 1
    A_eq.append(row); b_eq.append(period.production_kwh)

    # 7) fixed consumption
    row = np.zeros(len(C)); row[C.CONS] = 1
    A_eq.append(row); b_eq.append(period.consumption_kwh)

    A_eq = np.vstack(A_eq)
    b_eq = np.array(b_eq)

    # ------------------------------------------------------------------ #
    # Inequality constraints  A_ub · x  ≤ b_ub                            #
    # ------------------------------------------------------------------ #
    A_ub, b_ub = [], []

    # 1) charge limit   −ESS ≤ charge_cap  (ESS negative == charging)
    row = np.zeros(len(C)); row[C.ESS] = -1
    A_ub.append(row); b_ub.append(charge_cap)

    # 2) discharge limit  ESS ≤ discharge_cap
    row = np.zeros(len(C)); row[C.ESS] = 1
    A_ub.append(row); b_ub.append(discharge_cap)

    # grid import/export power limits can be enforced in a more complex
    # model; omitted here (same default as Java if not configured)

    A_ub = np.vstack(A_ub)
    b_ub = np.array(b_ub)

    # ------------------------------------------------------------------ #
    # Variable bounds                                                    #
    # (ESS and GRID allowed ±inf, everything else ≥ 0)                   #
    # ------------------------------------------------------------------ #
    bounds: Tuple[Tuple[float | None, float | None], ...] = tuple(
        (None, None) if idx in (C.ESS, C.GRID, C.GRID_TO_ESS) else (0, None)
        for idx in range(len(C))
    )

    # ------------------------------------------------------------------ #
    # Solve LP                                                           #
    # ------------------------------------------------------------------ #
    res = linprog(
        c, A_ub, b_ub, A_eq, b_eq, bounds=bounds, method="highs", options={"disp": False}
    )

    violations = 0 if res.success else 1
    if not res.success:
        # fall‑back: import everything from grid
        grid_to_cons = period.consumption_kwh - period.production_kwh
        cost = max(0, grid_to_cons) * period.price_buy
        revenue = 0
        return EnergyFlowResult(
            prod_to_ess=0,
            prod_to_cons=period.production_kwh,
            prod_to_grid=0,
            ess_to_cons=0,
            grid_to_ess=0,
            grid_to_cons=max(0, grid_to_cons),
            ess_net=0,
            grid_net=max(0, grid_to_cons),
            grid_buy_cost=cost,
            grid_sell_revenue=revenue,
            violated_constraints=1,
        )

    x = res.x.round(6)  # avoid tiny negatives

    # derive KPIs
    grid_charge = max(0, x[C.GRID_TO_ESS])
    batt_to_grid = -min(0, x[C.GRID_TO_ESS])
    grid_buy_cost = (x[C.GRID_TO_CONS] + grid_charge) * period.price_buy
    grid_buy_cost *= risk_factor if grid_charge > 0 else 1.0
    grid_sell_revenue = (x[C.PROD_TO_GRID] + batt_to_grid) * period.price_sell

    return EnergyFlowResult(
        prod_to_ess=x[C.PROD_TO_ESS],
        prod_to_cons=x[C.PROD_TO_CONS],
        prod_to_grid=x[C.PROD_TO_GRID],
        ess_to_cons=x[C.ESS_TO_CONS],
        grid_to_ess=x[C.GRID_TO_ESS],
        grid_to_cons=x[C.GRID_TO_CONS],
        ess_net=x[C.ESS],
        grid_net=x[C.GRID],
        grid_buy_cost=grid_buy_cost,
        grid_sell_revenue=grid_sell_revenue,
        violated_constraints=violations,
    )
