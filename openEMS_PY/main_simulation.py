# ----------------------------------------------------------------------
#  main_simulation.py  – roll‑out + KPI/plots in one place
# ----------------------------------------------------------------------
"""
Entry point that mirrors the OpenEMS Scheduler cycle:
    load → build GlobalOptimizationContext → simulate → KPIs → plots
Requires the helper modules already created (risk_level.py, simulator.py …).
"""

from __future__ import annotations

import time
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pvlib                                             # <- your PV fetch

# --- OpenEMS‑style helper modules --------------------------------------
from .system_parameters import SystemParameters
from .risk_level import RiskLevel
from .global_optimization_context import GlobalOptimizationContext
from .simulator import Simulator, SimulationResult        # noqa: F401

# ----------------------------------------------------------------------
# 0)  GLOBAL CONSTANTS  (reuse yours – change once here)                #
# ----------------------------------------------------------------------
YEAR              = 2021
ANNUAL_CONS_KWH   = 3_000
TZ                = "Europe/Berlin"
GRID_TAX_EUR_PER_KWH = 0.15
VAT_RATE             = 0.21

DATA_DIR   = Path(r"C:\Users\AlexB\OneDrive\Desktop\HeatUnicorns\MVP\Trial_C_Integration\OptimizationDynamicTarifs\GeneticAlgorithm_openEMS\CloneStructure\data")
LOAD_FILE  = DATA_DIR / "LoadProfilesSummary.csv"
PRICE_FILE = DATA_DIR / "DE_day_ahead_2023.csv"
PV_FILE    = DATA_DIR / "PV_Production.csv"              # (kept for clarity)

# ----------------------------------------------------------------------
# 1)  DATA LOADING  (your original code – almost unchanged)             #
# ----------------------------------------------------------------------
def _to_utc(series: pd.Series) -> pd.Series:
    """Parse anything datetime-ish to tz-aware UTC."""
    return pd.to_datetime(series, utc=True, errors="coerce")

def load_input_dataframe() -> pd.DataFrame:
    print("1. Loading and preprocessing data…")

    # ── 1. LOAD PROFILE
    load_df = pd.read_csv(LOAD_FILE, parse_dates=["Date"])
    load_df["Date"] = _to_utc(load_df["Date"])
    load_15min = (
        load_df.set_index("Date")
        .tz_convert(TZ)
        .loc[f"{YEAR}-01-01":f"{YEAR}-12-31 23:45", "Load"]
    )
    load_15min *= ANNUAL_CONS_KWH / load_15min.sum()

    # ── 2. PV PRODUCTION  – request hourly PVGIS → interpolate 15 min
    power_dc_df, *_ = pvlib.iotools.get_pvgis_hourly(
        latitude=51.176164, longitude=6.819423,
        start=YEAR, end=YEAR,
        raddatabase="PVGIS-SARAH3", pvcalculation=True,
        peakpower=6, pvtechchoice="crystSi",
        mountingplace="building", loss=14,
        surface_tilt=30, surface_azimuth=180,
        components=False, trackingtype=0, map_variables=True,
    )
    pv_hourly_kW = (power_dc_df["P"] / 1_000).tz_convert(TZ)
    pv_15min = (
        pv_hourly_kW
        .resample("15min", closed="left", label="left")
        .ffill()
        .reindex(load_15min.index, method="ffill")
        * 0.25                                     # kW → kWh/15 min
    )

    # ── 3. DAY‑AHEAD PRICES
    shift_years = 2023 - YEAR
    price_raw = pd.read_csv(PRICE_FILE, usecols=["timestamp", "price"])
    price_raw["timestamp"] = pd.to_datetime(price_raw["timestamp"], utc=True, errors="coerce")
    spot_h = (price_raw.set_index("timestamp")["price"] / 1000)          # €/kWh
    spot_h.index = spot_h.index - pd.DateOffset(years=shift_years)
    spot_h = spot_h.tz_convert(TZ)

    prices_hourly = spot_h.reindex(
        pd.date_range(f"{YEAR}-01-01 00:00", f"{YEAR}-12-31 23:00",
                      freq="1h", tz=TZ),
        method="ffill",
    )
    price_15min = prices_hourly.resample("15min").ffill()
    price_sell_15min = price_15min.clip(lower=0)
    price_buy_15min  = (price_15min + GRID_TAX_EUR_PER_KWH) * (1 + VAT_RATE)

    # quick sanity check
    print(f"   ▸ price std‑dev: {price_15min.std():.4f} €/kWh")

    # ── 4. ENSURE COMMON TIME SPAN
    ts0 = max(pv_15min.index.min(), load_15min.index.min(), price_15min.index.min())
    ts1 = min(pv_15min.index.max(), load_15min.index.max(), price_15min.index.max())

    pv_15min_kw   = pv_15min.loc[ts0:ts1] / 0.25
    load_15min_kw = load_15min.loc[ts0:ts1] / 0.25
    price_15min   = price_15min.loc[ts0:ts1]

    combined_df = pd.DataFrame({
        "load_kw"                  : load_15min_kw,
        "pv_kw"                    : pv_15min_kw,
        "price_buy_eur_per_kwh"    : price_buy_15min,
        "price_sell_eur_per_kwh"   : price_sell_15min,
    }).dropna()

    print("   ✔ data loading complete.")
    return combined_df

# ----------------------------------------------------------------------
# 2)  ROLLING‑HORIZON LOOP  (unchanged)                                #
# ----------------------------------------------------------------------
def run_simulation(df_input: pd.DataFrame,
                   system: SystemParameters,
                   risk: RiskLevel = RiskLevel.MEDIUM,
                   quarter_horizon: int = 24,
                   hour_horizon: int = 18) -> pd.DataFrame:

    soc = system.capacity_kwh * 0.50
    results: list[SimulationResult] = []

    max_start = len(df_input) - (quarter_horizon + hour_horizon * 4)
    t0 = time.time()
    for idx in tqdm(range(max_start), desc="simulate year"):

        ctx = GlobalOptimizationContext.from_dataframe(
            df_input, idx, system, soc, risk,
            quarter_horizon=quarter_horizon, hour_horizon=hour_horizon
        )
        res = Simulator.simulate(ctx)
        results.append(res)

        soc_delta = res.ess_net_kwh
        soc -= (soc_delta / system.discharge_eff) if soc_delta > 0 else (soc_delta * system.charge_eff)
        soc = np.clip(soc, system.min_soc_kwh, system.max_soc_kwh)

    print(f"\nTotal run‑time: {time.time()-t0:,.1f} s")
    return pd.DataFrame(r.as_dict() for r in results).set_index("time")

# ----------------------------------------------------------------------
# 3)  KPI PRINT + PLOTS                                                #
# ----------------------------------------------------------------------
def kpi_summary(df: pd.DataFrame, system: SystemParameters) -> None:
    baseline_cost = (
        (df.load_kwh - df.pv_kwh)
        .mul(np.where(df.load_kwh >= df.pv_kwh,
                      df.price_buy_eur_per_kwh,
                      df.price_sell_eur_per_kwh))
        .sum()
    )
    print("\n--- KPI ----------------------------------------------------")
    print(f"Baseline (PV only) cost     : {baseline_cost:10.2f} €")
    print(f"Optimised battery cost       : {df.cost_eur.sum():10.2f} €")
    print(f"→ Savings                    : {baseline_cost-df.cost_eur.sum():10.2f} €")
    print("Battery mode share:")
    print(df.battery_mode.value_counts(normalize=True).mul(100).round(1).astype(str) + " %")
    print("-----------------------------------------------------------")

def plot_first_day(df: pd.DataFrame):
    day0 = df.index[0].date().isoformat()
    start, end = pd.Timestamp(day0, tz=df.index.tz), pd.Timestamp(day0, tz=df.index.tz) + timedelta(days=1)
    d = df[start:end]
    if d.empty:
        print("Plot skipped – no data"); return

    fig, ax1 = plt.subplots(figsize=(12,6))
    ax1.set_title(f"Energy flows – {day0}")
    ax1.plot(d.index, d.pv_kwh, label="PV kWh", color="orange")
    ax1.plot(d.index, d.load_kwh, label="Load kWh", color="steelblue")
    ax1.fill_between(d.index, d.pv_to_ess, label="PV→ESS", alpha=.3)
    ax1.fill_between(d.index, -d.ess_to_load, label="ESS→Load", alpha=.3)
    ax1.set_ylabel("kWh per 15 min")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(d.index, d.price_buy_eur_per_kwh, color="red", label="Buy €/kWh", linewidth=.8)
    ax2.set_ylabel("€/kWh")
    fig.tight_layout(); plt.show()

# ----------------------------------------------------------------------
# 4)  MAIN                                                             #
# ----------------------------------------------------------------------
if __name__ == "__main__":
    df_raw  = load_input_dataframe()
    system  = SystemParameters(capacity_kwh=5)

    df_res  = run_simulation(df_raw, system, risk=RiskLevel.MEDIUM)
    kpi_summary(df_res, system)
    plot_first_day(df_res)


