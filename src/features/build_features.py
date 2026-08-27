"""Pipeline de feature engineering temporal para el pronóstico del SEN, con Polars.

Los lags/rolling/ciclos se calculan **por nodo** (`.over("node")`): el dataset es
un panel de 5 barras, y mezclar nodos dentro de una misma ventana temporal
mezclaría series sin relación causal directa entre sí, además de romper la
continuidad horaria que lags/rolling asumen.
"""
from __future__ import annotations

import math
from pathlib import Path

import holidays
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "sen_hourly_data.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "sen_features.parquet"

TARGET_COLUMNS = ["solar_generation_mwh", "wind_generation_mwh", "demand_mwh", "marginal_cost_usd_mwh"]
LAG_HOURS = [1, 24, 168]
ROLLING_WINDOWS = [6, 24]

# Radiación solar (GHI), velocidad de viento y temperatura: los drivers
# meteorológicos reales de la generación renovable (ver
# `src/data/fetch_energy_data.py`, donde solar/eólica se derivan de estas
# variables, no al revés). Lags más cortos que los de los targets
# (`LAG_HOURS`): el clima se autocorrelaciona y decae más rápido hora a hora
# que un patrón semanal de demanda, así que 1h/3h/6h capturan la persistencia
# de corto plazo (nubosidad, un frente de viento) y 24h captura el ciclo
# diurno (misma hora, un día atrás) -- no hace falta un lag de 168h como en
# los targets, porque el clima no tiene un patrón semanal propio.
WEATHER_COLUMNS = ["ghi_w_m2", "wind_speed_ms", "temperature_c"]
WEATHER_LAG_HOURS = [1, 3, 6, 24]

TWO_PI = 2 * math.pi
CHILE_HOLIDAYS = holidays.Chile(years=range(2023, 2028))


def load_raw_data(path: Path = RAW_PATH) -> pl.DataFrame:
    return pl.read_csv(path, try_parse_dates=True)


def add_lag_features(df: pl.DataFrame, columns: list[str] = TARGET_COLUMNS, lags: list[int] = LAG_HOURS) -> pl.DataFrame:
    """Agrega `<col>_lag_<h>h` -- el valor de `col` hace `h` horas, en ese mismo nodo."""
    df = df.sort(["node", "timestamp"])
    exprs = [
        pl.col(col).shift(lag).over("node").alias(f"{col}_lag_{lag}h")
        for col in columns
        for lag in lags
    ]
    return df.with_columns(exprs)


def add_weather_lag_features(
    df: pl.DataFrame, columns: list[str] = WEATHER_COLUMNS, lags: list[int] = WEATHER_LAG_HOURS
) -> pl.DataFrame:
    """Agrega `<col>_lag_<h>h` para las variables meteorológicas, por nodo.

    Deliberadamente **lags, nunca el valor crudo de la hora actual**: aunque
    la irradiancia/viento/temperatura de la hora que se está pronosticando
    sí podrían conocerse en producción vía un pronóstico NWP (a diferencia de
    un target, que nunca se conoce por adelantado), este proyecto no simula
    pronósticos meteorológicos -- solo observaciones históricas -- así que
    usar el valor de la hora actual sería, en la práctica, filtración: el
    dataset no distingue "clima observado" de "clima que en verdad se sabría
    de antemano". Los lags evitan ese problema sin necesitar esa distinción.
    """
    df = df.sort(["node", "timestamp"])
    exprs = [
        pl.col(col).shift(lag).over("node").alias(f"{col}_lag_{lag}h")
        for col in columns
        for lag in lags
    ]
    return df.with_columns(exprs)


def add_rolling_features(df: pl.DataFrame, columns: list[str] = TARGET_COLUMNS, windows: list[int] = ROLLING_WINDOWS) -> pl.DataFrame:
    """Agrega media y desvío móvil `<col>_rolling_{mean,std}_<w>h`, por nodo.

    Se calculan sobre `shift(1)` (la hora anterior hacia atrás) para que la
    ventana nunca incluya la propia fila -- de lo contrario el feature vería el
    valor que se está intentando predecir.
    """
    df = df.sort(["node", "timestamp"])
    exprs = []
    for col in columns:
        shifted = pl.col(col).shift(1)
        for w in windows:
            exprs.append(shifted.rolling_mean(window_size=w).over("node").alias(f"{col}_rolling_mean_{w}h"))
            exprs.append(shifted.rolling_std(window_size=w).over("node").alias(f"{col}_rolling_std_{w}h"))
    return df.with_columns(exprs)


def add_cyclical_features(df: pl.DataFrame) -> pl.DataFrame:
    """Codificación seno/coseno de hora del día (periodo 24) y mes (periodo 12):
    evita el salto artificial 23->0 / dic->ene que tendría un entero directo."""
    hour = pl.col("timestamp").dt.hour()
    month = pl.col("timestamp").dt.month()
    return df.with_columns([
        (hour * (TWO_PI / 24)).sin().alias("hour_sin"),
        (hour * (TWO_PI / 24)).cos().alias("hour_cos"),
        ((month - 1) * (TWO_PI / 12)).sin().alias("month_sin"),
        ((month - 1) * (TWO_PI / 12)).cos().alias("month_cos"),
    ])


def add_calendar_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        pl.col("timestamp").dt.weekday().alias("day_of_week"),
        (pl.col("timestamp").dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
    ])


def add_holiday_feature(df: pl.DataFrame) -> pl.DataFrame:
    """Marca feriado legal chileno, vía la librería `holidays` (calcula
    correctamente los feriados movibles -- Viernes/Sábado Santo, etc.)."""
    dates = df.get_column("timestamp").dt.date().to_list()
    is_holiday = [d in CHILE_HOLIDAYS for d in dates]
    return df.with_columns(pl.Series("is_holiday", is_holiday, dtype=pl.Int8))


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    """Pipeline completo de feature engineering sobre el panel horario crudo."""
    df = add_lag_features(df)
    df = add_weather_lag_features(df)
    df = add_rolling_features(df)
    df = add_cyclical_features(df)
    df = add_calendar_features(df)
    df = add_holiday_feature(df)
    return df


if __name__ == "__main__":
    raw = load_raw_data()
    features = build_features(raw)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.write_parquet(PROCESSED_PATH)

    print(f"Features generados: {features.shape[0]:,} filas x {features.shape[1]} columnas")
    print(f"Guardado en {PROCESSED_PATH}")
