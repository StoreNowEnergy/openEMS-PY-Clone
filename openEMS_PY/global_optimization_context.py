"""
global_optimization_context.py
==============================

Python mirror of
  io.openems.edge.energy.api.simulation.GlobalOptimizationContext

The *RiskLevel* enum lives in a **separate** module `risk_level.py`
(as requested).  Everything else – PeriodDuration, Period, Ess, Grid,
the helper methods and factory – is embedded here for 1‑to‑1 traceability.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Sequence

import pandas as pd

from risk_level import RiskLevel  # <- kept in its own file


# ---------------------------------------------------------------------------#
#  PeriodDuration (inner enum in Java)                                       #
# ---------------------------------------------------------------------------#

class PeriodDuration(str, Enum):
    QUARTER = "QUARTER"        # 15‑minute slice
    HOUR = "HOUR"              # 60‑minute slice

    @property
    def hours(self) -> float:
        return 0.25 if self is PeriodDuration.QUARTER else 1.0

    @property
    def kwh_to_kw(self) -> float:
        return 1.0 / self.hours

    @property
    def kw_to_kwh(self) -> float:
        return self.hours


# ---------------------------------------------------------------------------#
#  Period (inner static class in Java)                                       #
# ---------------------------------------------------------------------------#

@dataclass(slots=True)
class Period:
    index: int
    time: datetime
    duration: PeriodDuration
    production_kwh: float
    consumption_kwh: float
    price_buy: float    # €/kWh
    price_sell: float   # €/kWh

    # conversion helpers identical to Java’s getPower()/getEnergy()
    def energy_to_power(self, energy_kwh: float) -> float:
        return energy_kwh * self.duration.kwh_to_kw

    def power_to_energy(self, power_kw: float) -> float:
        return power_kw * self.duration.kw_to_kwh


# ---------------------------------------------------------------------------#
#  Ess + Grid (inner static classes in Java)                                 #
# ---------------------------------------------------------------------------#

@dataclass(slots=True)
class Ess:
    capacity_kwh: float = 5.0
    max_charge_kw: float = 3.68
    max_discharge_kw: float = 3.68
    charge_eff: float = 0.95
    discharge_eff: float = 0.95
    min_soc_pct: float = 0.15
    max_soc_pct: float = 0.90

    @property
    def min_soc_kwh(self) -> float:
        return self.capacity_kwh * self.min_soc_pct

    @property
    def max_soc_kwh(self) -> float:
        return self.capacity_kwh * self.max_soc_pct


@dataclass(slots=True)
class Grid:
    max_buy_kw: float = 10.0
    max_sell_kw: float = 6.0


# ---------------------------------------------------------------------------#
#  GlobalOptimizationContext (outer class in Java)                           #
# ---------------------------------------------------------------------------#

@dataclass(slots=True)
class GlobalOptimizationContext:
    periods: Sequence[Period]
    ess: Ess
    grid: Grid
    initial_soc_kwh: float
    risk_level: RiskLevel = RiskLevel.MEDIUM

    # ------------------------------------------------------------------ #
    # Convenience properties                                             #
    # ------------------------------------------------------------------ #

    @property
    def risk_factor(self) -> float:
        """Numeric multiplier identical to Java’s RiskLevel.efficiencyFactor."""
        return self.risk_level.efficiency_factor

    def as_dataframe(self) -> pd.DataFrame:
        """Return the period list as a tidy DataFrame (good for KPIs/plots)."""
        return pd.DataFrame(
            {
                "index": [p.index for p in self.periods],
                "time": [p.time for p in self.periods],
                "duration": [p.duration.value for p in self.periods],
                "production_kwh": [p.production_kwh for p in self.periods],
                "consumption_kwh": [p.consumption_kwh for p in self.periods],
                "price_buy": [p.price_buy for p in self.periods],
                "price_sell": [p.price_sell for p in self.periods],
            }
        ).set_index("index")

    # ------------------------------------------------------------------ #
    # Static helpers ported verbatim                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_period_duration_hour_from_index(time: datetime) -> int:
        """
        Java logic:

            index = 6 h (24 quarters) + remaining quarters of the running hour
        """
        minute = 60 if time.minute == 0 else time.minute
        return 6 * 4 + (60 - minute) // 15

    # ------------------------------------------------------------------ #
    # Factory – default 24 h horizon (6 h quarters + 18 h hours)          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame,
        start_idx: int,
        ess: Ess,
        grid: Grid,
        initial_soc_kwh: float,
        risk_level: RiskLevel = RiskLevel.MEDIUM,
        quarter_horizon: int = 24,
        hour_horizon: int = 18,
    ) -> "GlobalOptimizationContext":
        """
        Build a horizon identical to the Java default:

            24 × QUARTER  +  18 × HOUR  = 24 hours
        """
        periods: List[Period] = []

        # ---- 15‑minute quarters ----------------------------------------
        for i in range(quarter_horizon):
            row_idx = start_idx + i
            if row_idx >= len(df):
                break
            row = df.iloc[row_idx]
            periods.append(
                Period(
                    index=i,
                    time=row.name,
                    duration=PeriodDuration.QUARTER,
                    production_kwh=row.pv_kw * 0.25,
                    consumption_kwh=row.load_kw * 0.25,
                    price_buy=row.price_buy_eur_per_kwh,
                    price_sell=row.price_sell_eur_per_kwh,
                )
            )

        # ---- hourly aggregates ----------------------------------------
        for j in range(hour_horizon):
            base = start_idx + quarter_horizon + j * 4
            slice_ = df.iloc[base : base + 4]
            if slice_.empty:
                break
            periods.append(
                Period(
                    index=quarter_horizon + j,
                    time=slice_.index[0],
                    duration=PeriodDuration.HOUR,
                    production_kwh=(slice_.pv_kw * 0.25).sum(),
                    consumption_kwh=(slice_.load_kw * 0.25).sum(),
                    price_buy=slice_.price_buy_eur_per_kwh.mean(),
                    price_sell=slice_.price_sell_eur_per_kwh.mean(),
                )
            )

        return GlobalOptimizationContext(
            periods=periods,
            ess=ess,
            grid=grid,
            initial_soc_kwh=initial_soc_kwh,
            risk_level=risk_level,
        )
