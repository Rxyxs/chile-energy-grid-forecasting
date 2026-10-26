"""Panel horario REAL del SEN: 6 barras de 220 kV + el sistema completo, 2022-2025.

Todas las series vienen de fuentes públicas reales:

- **Costo marginal horario por barra** (USD/MWh), reconstruido de los gráficos
  vectoriales de los informes diarios del Coordinador (``cen_daily_reports``):
  Crucero, Cardones, Pan de Azúcar, Quillota, Charrúa y Puerto Montt.
- **Generación horaria del SEN por tecnología** (MW), del mismo informe: la
  solar y la eólica son los targets de sistema, y la suma de las cinco
  tecnologías (térmica, hidráulica, solar, eólica y baterías) es la generación
  bruta total, el indicador público más cercano a la demanda horaria del
  sistema (en un sistema aislado, generación = demanda + pérdidas).
- **Clima horario** en las coordenadas de cada barra (Open-Meteo, ERA5).

Ninguna fuente pública trae generación ni demanda *por barra*, así que el
panel no las inventa: tiene un nodo más, ``SEN``, que lleva los targets de
sistema (solar, eólica y demanda) con el clima promedio de las 6 barras. Las
filas de cada barra también llevan esas series del sistema, para que sus
rezagos entren como variables del precio (el orden de mérito depende de la
demanda residual del sistema, no de la de la barra). ``rows_for_target``
define qué filas se entrenan y se evalúan para cada target, así ninguna
métrica cuenta 7 veces la misma hora del sistema.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import cen_daily_reports as reports
from src.data import real_sources

START = "2022-01-01 00:00"
END = "2025-12-31 23:00"

SYSTEM_NODE = "SEN"
BAR_NODES = {
    "Crucero": "Crucero_220kV",
    "Cardones": "Cardones_220kV",
    "Pan de Azucar": "Pan_de_Azucar_220kV",
    "Quillota": "Quillota_220kV",
    "Charrua": "Charrua_220kV",
    "Puerto Montt": "Puerto_Montt_220kV",
}
NODES = [*BAR_NODES.values(), SYSTEM_NODE]

PRICE_TARGET = "marginal_cost_usd_mwh"
SYSTEM_TARGETS = ["solar_generation_mwh", "wind_generation_mwh", "demand_mwh"]
WEATHER_PATH = real_sources.WEATHER_PATH
TECH_COLUMNS = ["termica", "hidraulica", "solar", "eolica", "bess"]
ANOMALY_SHARE = 0.5  # una hora bajo la mitad de la mediana de su día no es demanda real


def rows_for_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Filas que se entrenan y evalúan para ``target``: el nodo SEN para los
    targets de sistema, las 6 barras para el costo marginal."""
    if "node" not in df.columns or SYSTEM_NODE not in set(df["node"].unique()):
        return df  # panel simulado (use_real=False): cada barra tiene sus 4 targets
    mask = df["node"].eq(SYSTEM_NODE) if target in SYSTEM_TARGETS else df["node"].ne(SYSTEM_NODE)
    return df[mask]


def _hour_timestamps(frame: pd.DataFrame) -> pd.Series:
    # La hora h del informe (1-24) es el bloque que termina a las h:00 hora local.
    return pd.to_datetime(frame["date"]) + pd.to_timedelta(frame["hour"] - 1, unit="h")


def load_system_hourly() -> pd.DataFrame:
    """Solar, eólica y generación total del SEN por hora (MW = MWh en una hora)."""
    gen = pd.read_csv(reports.GEN_HOURLY_PATH)
    for col in TECH_COLUMNS:
        if col not in gen:
            gen[col] = 0.0
    # Una tecnología sin área en el gráfico ese día (baterías en 2022, solar de
    # noche) generó 0, no un dato faltante.
    gen[TECH_COLUMNS] = gen[TECH_COLUMNS].fillna(0.0)
    out = pd.DataFrame({
        "timestamp": _hour_timestamps(gen),
        "solar_generation_mwh": gen["solar"],
        "wind_generation_mwh": gen["eolica"],
        "demand_mwh": gen[TECH_COLUMNS].sum(axis=1),
    })
    # Horas que no son demanda real: la hora que no existe el día que empieza
    # el horario de verano (el gráfico la dibuja en 0) y el apagón nacional del
    # 25 de febrero de 2025. Quedan como faltantes, fuera de entrenamiento y métricas.
    daily_median = out.groupby(gen["date"])["demand_mwh"].transform("median")
    out.loc[out["demand_mwh"] < ANOMALY_SHARE * daily_median, SYSTEM_TARGETS] = np.nan
    return out


def load_bar_cmg() -> pd.DataFrame:
    cmg = pd.read_csv(reports.CMG_HOURLY_PATH)
    cmg = cmg[cmg["bar"].isin(BAR_NODES)]
    return pd.DataFrame({
        "timestamp": _hour_timestamps(cmg),
        "node": cmg["bar"].map(BAR_NODES),
        PRICE_TARGET: cmg["cmg_usd_mwh"].clip(lower=0.0),
        "cmg_kind": cmg["kind"],
    })


def build_real_panel(start: str = START, end: str = END) -> pd.DataFrame:
    grid = pd.date_range(start, end, freq="h")
    system = load_system_hourly().set_index("timestamp").reindex(grid)
    weather = pd.read_csv(WEATHER_PATH, parse_dates=["timestamp"])
    cmg = load_bar_cmg()

    frames = []
    for node in BAR_NODES.values():
        w = weather[weather["node"] == node].set_index("timestamp").reindex(grid)
        price = cmg[cmg["node"] == node].set_index("timestamp")[PRICE_TARGET].reindex(grid)
        frames.append(pd.DataFrame({
            "timestamp": grid, "node": node, **{c: system[c].to_numpy() for c in SYSTEM_TARGETS},
            PRICE_TARGET: price.to_numpy(),
            **{c: w[c].to_numpy() for c in ("ghi_w_m2", "wind_speed_ms", "temperature_c")},
        }))
    bars = pd.concat(frames, ignore_index=True)

    # El nodo SEN: targets de sistema, clima promedio de las barras y, como
    # variable (no como target), el costo marginal promedio de las barras.
    sen = bars.groupby("timestamp", as_index=False)[[PRICE_TARGET, "ghi_w_m2", "wind_speed_ms", "temperature_c"]].mean()
    sen = sen.assign(node=SYSTEM_NODE, **{c: system[c].reindex(sen["timestamp"]).to_numpy() for c in SYSTEM_TARGETS})
    panel = pd.concat([bars, sen[bars.columns]], ignore_index=True)
    rounding = {c: 2 for c in [*SYSTEM_TARGETS, PRICE_TARGET, "wind_speed_ms", "temperature_c"]}
    return panel.round({**rounding, "ghi_w_m2": 1}).sort_values(["node", "timestamp"]).reset_index(drop=True)


def cmg_kind_by_date() -> pd.Series:
    """Qué gráfico de costo marginal trae cada día (real preliminar o en línea)."""
    cmg = pd.read_csv(reports.CMG_HOURLY_PATH, usecols=["date", "kind"]).drop_duplicates("date")
    return cmg.set_index("date")["kind"]


if __name__ == "__main__":
    p = build_real_panel()
    print(p.groupby("node")[[*SYSTEM_TARGETS, PRICE_TARGET]].count())
    print(p.describe().T.round(1))
    print(np.round(p.isna().mean(), 3))
