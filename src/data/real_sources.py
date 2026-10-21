"""Fuentes reales que anclan el panel horario del SEN.

El Coordinador Eléctrico Nacional publica el costo marginal horario por barra,
pero su API exige una cuenta institucional y su sitio bloquea descargas
automatizadas. Este módulo reemplaza todo lo que sí se puede obtener de fuentes
públicas reales:

- **Clima horario real** (Open-Meteo, reanálisis ERA5) en las coordenadas de
  las 5 barras: irradiancia global horizontal, viento a 100 m y temperatura.
  Reemplaza al clima simulado; la generación solar y eólica se derivan de él.
- **Generación mensual real del SEN** por tecnología (datos.gob.cl, CNE): la
  forma mensual de la generación solar, eólica y total calibra la solar, la
  eólica y la demanda del panel.
- **Costo marginal mensual real de la barra Crucero 220 kV** (reportes
  mensuales de la CNE, extraídos en el gemelo SAG de este portafolio): fija el
  nivel mensual del precio.

Las tres series se guardan en ``data/real/`` y se versionan, así que el
pipeline y los tests corren sin red; ``refresh_all()`` las vuelve a descargar.
"""
from __future__ import annotations

import io
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_DIR = PROJECT_ROOT / "data" / "real"
WEATHER_PATH = REAL_DIR / "weather_hourly_open_meteo.csv.gz"
GENERATION_PATH = REAL_DIR / "sen_generacion_mensual.csv"
CMG_PATH = REAL_DIR / "cne_crucero_cmg_mensual.csv"

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
GENERATION_URL = ("https://datos.gob.cl/dataset/0e54541d-e5b4-47a5-9cc8-0bd5062d1c88/resource/"
                  "389a1943-9c3d-4957-982a-58e3fb0c1bdb/download/generacion-bruta-mensual-sen.csv")

# Coordenadas aproximadas de cada subestación (lat, lon).
NODE_COORDS: dict[str, tuple[float, float]] = {
    "Crucero_220kV": (-22.27, -69.57),      # María Elena, Antofagasta
    "Cardones_220kV": (-27.39, -70.33),     # Copiapó, Atacama
    "Quillota_220kV": (-32.88, -71.25),     # Quillota, Valparaíso
    "Alto_Jahuel_220kV": (-33.73, -70.68),  # Buin, Metropolitana
    "Charrua_220kV": (-37.13, -72.33),      # Cabrero, Biobío
}

SOLAR_TECH = ("Solar",)
WIND_TECH = ("Eólica",)


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception:  # pragma: no cover - red
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def fetch_weather(start: str, end: str) -> pd.DataFrame:
    """Clima horario real por barra, en hora local de Chile (America/Santiago).

    Los cambios de horario dejan una hora repetida y una faltante por año; la
    repetida se promedia y la faltante se interpola, para quedar en una grilla
    horaria regular igual a la del panel."""
    frames = []
    for node, (lat, lon) in NODE_COORDS.items():
        url = (f"{OPEN_METEO_URL}?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
               "&hourly=shortwave_radiation,wind_speed_100m,temperature_2m"
               "&wind_speed_unit=ms&timezone=America%2FSantiago")
        data = json.loads(_get(url))["hourly"]
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(data["time"]),
            "ghi_w_m2": data["shortwave_radiation"],
            "wind_speed_ms": data["wind_speed_100m"],
            "temperature_c": data["temperature_2m"],
        }).groupby("timestamp", as_index=False).mean()
        grid = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
        df = df.set_index("timestamp").reindex(grid).interpolate(limit=3).rename_axis("timestamp").reset_index()
        df.insert(1, "node", node)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def fetch_generation() -> pd.DataFrame:
    """Generación bruta mensual real del SEN (MWh): total, solar y eólica."""
    raw = pd.read_csv(io.BytesIO(_get(GENERATION_URL)), dtype=str)
    raw["generacion_mwh"] = pd.to_numeric(raw["generacion_mwh"].str.replace(".", "", regex=False)
                                          .str.replace(",", ".", regex=False), errors="coerce")
    raw["month"] = raw["anio"].str.zfill(4) + "-" + raw["mes"].str.zfill(2)
    tech = raw["tecnologia"].fillna("")
    out = pd.DataFrame({
        "total_mwh": raw.groupby("month")["generacion_mwh"].sum(),
        "solar_mwh": raw[tech.str.startswith(SOLAR_TECH)].groupby("month")["generacion_mwh"].sum(),
        "wind_mwh": raw[tech.str.startswith(WIND_TECH)].groupby("month")["generacion_mwh"].sum(),
    }).fillna(0.0)
    return out.reset_index().rename(columns={"index": "month"})


def refresh_all(start: str = "2024-01-01", end: str = "2025-12-31") -> None:
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    fetch_weather(start, end).round(2).to_csv(WEATHER_PATH, index=False, compression="gzip")
    fetch_generation().round(1).to_csv(GENERATION_PATH, index=False)


def load_weather() -> pd.DataFrame:
    return pd.read_csv(WEATHER_PATH, parse_dates=["timestamp"])


def load_generation() -> pd.DataFrame:
    return pd.read_csv(GENERATION_PATH, dtype={"month": str})


def load_cmg() -> pd.DataFrame:
    return pd.read_csv(CMG_PATH, dtype={"month": str})


def monthly_shape(values: pd.Series) -> pd.Series:
    """Serie mensual normalizada a media 1 (forma estacional, sin nivel)."""
    return values / values.mean()


def calibration_factors(sim_monthly: pd.Series, real_monthly: pd.Series) -> pd.Series:
    """Factor por mes que lleva la forma mensual simulada a la forma mensual real,
    conservando el nivel promedio simulado: f_m = forma_real_m / forma_sim_m."""
    common = sim_monthly.index.intersection(real_monthly.index)
    return monthly_shape(real_monthly.loc[common]) / monthly_shape(sim_monthly.loc[common])


if __name__ == "__main__":
    refresh_all()
    w, g = load_weather(), load_generation()
    print(f"clima: {len(w):,} filas, {w['node'].nunique()} barras, {w['timestamp'].min()} -> {w['timestamp'].max()}")
    print(f"generacion: {len(g)} meses, {g['month'].min()} -> {g['month'].max()}")
