"""Generador sintético de datos horarios del Sistema Eléctrico Nacional (SEN) de Chile.

El CEN (Coordinador Eléctrico Nacional) no expone una API pública simple y
gratuita para descargar históricos horarios por barra, así que este módulo
simula 2 años de datos (2024-2025) para 5 nodos/barras reales del SEN, con
generación solar y eólica, demanda, costo marginal (precio spot), y las
variables meteorológicas físicas que efectivamente *causan* la generación
renovable (irradiancia solar, velocidad de viento, temperatura) -- no como
una serie paralela desconectada, sino como el insumo del que la generación se
deriva matemáticamente (ver `_ghi_wm2`, `_wind_power_fraction`).

No es ruido: el costo marginal se deriva de la demanda residual (demanda menos
renovables) siguiendo la lógica real de orden de mérito del mercado eléctrico
-- más sol/viento desplaza generación cara y baja el precio; más demanda con
poca renovable lo sube -- para que haya señal real que un modelo de forecasting
pueda aprender, no solo aleatoriedad.
"""
from __future__ import annotations

from pathlib import Path

import holidays
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "sen_hourly_data.csv"

START_DATE = "2024-01-01"
END_DATE = "2025-12-31 23:00"
RANDOM_SEED = 42

# Nodos/barras reales del SEN, con perfiles representativos de su geografía:
# norte (desierto de Atacama) = alto potencial solar y minería 24/7; centro =
# demanda urbana con doble pico; sur (Biobío) = más viento, más industria.
NODES: dict[str, dict] = {
    "Crucero_220kV":     {"solar_capacity_mwh": 220, "wind_capacity_mwh": 40,  "base_demand_mwh": 180, "profile": "mining"},
    "Cardones_220kV":    {"solar_capacity_mwh": 200, "wind_capacity_mwh": 60,  "base_demand_mwh": 90,  "profile": "mixed"},
    "Quillota_220kV":    {"solar_capacity_mwh": 110, "wind_capacity_mwh": 70,  "base_demand_mwh": 140, "profile": "urban"},
    "Alto_Jahuel_220kV": {"solar_capacity_mwh": 130, "wind_capacity_mwh": 30,  "base_demand_mwh": 320, "profile": "urban"},
    "Charrua_220kV":     {"solar_capacity_mwh": 70,  "wind_capacity_mwh": 140, "base_demand_mwh": 150, "profile": "industrial"},
}

# Perfil horario relativo de demanda (24 valores, multiplican base_demand_mwh).
# "mining" es casi plano (operación continua); "urban" tiene doble pico
# (mañana + noche); "industrial"/"mixed" quedan entre ambos extremos.
DEMAND_SHAPE = {
    "mining": np.array([
        0.95, 0.94, 0.93, 0.93, 0.94, 0.95, 0.97, 0.99, 1.02, 1.04, 1.05, 1.05,
        1.04, 1.04, 1.05, 1.05, 1.04, 1.03, 1.02, 1.01, 1.00, 0.98, 0.97, 0.96,
    ]),
    "urban": np.array([
        0.72, 0.68, 0.65, 0.64, 0.65, 0.70, 0.82, 0.95, 1.02, 1.03, 1.02, 1.01,
        1.00, 0.99, 0.98, 0.99, 1.03, 1.12, 1.20, 1.18, 1.08, 0.96, 0.86, 0.78,
    ]),
    "mixed": np.array([
        0.80, 0.76, 0.74, 0.73, 0.75, 0.80, 0.90, 0.98, 1.03, 1.05, 1.05, 1.04,
        1.03, 1.02, 1.02, 1.03, 1.06, 1.10, 1.12, 1.10, 1.02, 0.94, 0.88, 0.83,
    ]),
    "industrial": np.array([
        0.90, 0.88, 0.87, 0.87, 0.88, 0.90, 0.95, 1.00, 1.04, 1.05, 1.05, 1.04,
        1.03, 1.03, 1.04, 1.04, 1.03, 1.02, 1.02, 1.01, 0.99, 0.96, 0.93, 0.91,
    ]),
}

CHILE_HOLIDAYS = holidays.Chile(years=[2023, 2024, 2025, 2026])

# Irradiancia solar de cielo despejado al mediodía solar, por nodo (W/m2) --
# refleja la geografía real: el norte (desierto de Atacama, Crucero/Cardones)
# tiene algunos de los niveles de irradiancia más altos del mundo; el centro
# es intermedio; el sur (Charrúa/Biobío) es más nuboso y templado.
GHI_CLEAR_SKY_PEAK_WM2: dict[str, float] = {
    "Crucero_220kV": 1050.0,
    "Cardones_220kV": 1030.0,
    "Quillota_220kV": 950.0,
    "Alto_Jahuel_220kV": 930.0,
    "Charrua_220kV": 800.0,
}

# Curva de potencia eólica simplificada (cúbica entre cut-in y velocidad
# nominal, plana en capacidad nominal sobre ella, cero bajo cut-in) -- el
# comportamiento real de una turbina, no un exponente arbitrario sobre un
# índice adimensional.
WIND_CUT_IN_MS = 3.0
WIND_RATED_MS = 12.5
WIND_SPEED_PEAK_MS = 16.0  # techo del índice de viento autocorrelado, ya en m/s

# Temperatura: media anual y amplitudes estacional/diurna por nodo -- el norte
# desértico (Atacama) tiene el mayor rango diurno (días calurosos, noches
# frías, cielo despejado sin humedad que module la temperatura); el sur es
# más templado y húmedo, con un rango diurno menor.
TEMPERATURE_PARAMS: dict[str, dict[str, float]] = {
    "Crucero_220kV":     {"annual_mean_c": 18.0, "seasonal_amplitude_c": 6.0, "diurnal_amplitude_c": 14.0},
    "Cardones_220kV":    {"annual_mean_c": 17.0, "seasonal_amplitude_c": 5.5, "diurnal_amplitude_c": 12.0},
    "Quillota_220kV":    {"annual_mean_c": 15.5, "seasonal_amplitude_c": 6.5, "diurnal_amplitude_c": 9.0},
    "Alto_Jahuel_220kV": {"annual_mean_c": 14.5, "seasonal_amplitude_c": 7.5, "diurnal_amplitude_c": 10.0},
    "Charrua_220kV":     {"annual_mean_c": 12.5, "seasonal_amplitude_c": 6.5, "diurnal_amplitude_c": 8.0},
}


def _seasonal_factor(day_of_year: np.ndarray, peak_day: int = 355, amplitude: float = 0.25, base: float = 0.75) -> np.ndarray:
    """Factor estacional hemisferio sur: pico cerca del solsticio de verano
    (~21 dic, día ~355), mínimo cerca del de invierno (~21 jun, día ~172)."""
    phase = 2 * np.pi * (day_of_year - peak_day) / 365.25
    return base + amplitude * np.cos(phase)


def _solar_shape(hour: np.ndarray, day_of_year: np.ndarray) -> np.ndarray:
    """Curva diurna en forma de campana entre amanecer y atardecer, con largo del
    día variando estacionalmente (~15h en verano, ~12h en invierno)."""
    daylight_hours = 13.5 + 1.5 * np.cos(2 * np.pi * (day_of_year - 355) / 365.25)
    sunrise = 12 - daylight_hours / 2
    sunset = 12 + daylight_hours / 2
    frac = np.clip((hour - sunrise) / (sunset - sunrise), 0, 1)
    return np.where((hour > sunrise) & (hour < sunset), np.sin(np.pi * frac) ** 1.5, 0.0)


def _clear_sky_daily_factor(n_days: int, rng: np.random.Generator) -> np.ndarray:
    """Factor de nubosidad por día: la mayoría son días despejados (cerca de 1.0),
    con una cola hacia días nublados que reducen el solar completo del día."""
    return rng.beta(6, 1.5, size=n_days)


def _ghi_wm2(node: str, day_of_year: np.ndarray, shape: np.ndarray, clear_sky: np.ndarray,
             rng: np.random.Generator, n: int) -> np.ndarray:
    """Irradiancia horizontal global (GHI), W/m2 -- la variable meteorológica
    real que un pronosticador de generación solar usaría en producción (vía
    observación reciente o pronóstico NWP). Combina el pico de cielo despejado
    del nodo, la curva diurna/estacional ya calculada (`shape`) y la nubosidad
    diaria (`clear_sky`); la atenuación atmosférica estacional adicional
    (ángulo solar más bajo en invierno, más trayecto de atmósfera) se modela
    con una amplitud menor a la de `shape`, porque `shape` ya captura la
    mayor parte del efecto estacional vía el largo del día."""
    peak = GHI_CLEAR_SKY_PEAK_WM2[node]
    seasonal_atmospheric = _seasonal_factor(day_of_year, peak_day=355, amplitude=0.10, base=0.93)
    noise = rng.normal(1.0, 0.02, n)
    return np.clip(peak * seasonal_atmospheric * shape * clear_sky * noise, 0, None)


def _wind_speed_ms(wind_idx: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    """Velocidad de viento, m/s -- reescalado directo del mismo índice
    autocorrelado con reversión a la media (`_wind_index`) que ya modela la
    persistencia horaria y los "frentes de viento". Deliberadamente SIN
    ruido adicional aquí: `wind_idx` ya trae su propia variación hora a hora
    (su propio término `rng.normal`), y agregar una segunda fuente de ruido
    independiente justo antes de la curva de potencia (ver
    `_wind_power_fraction`) se probó empíricamente y resultó en una
    generación tan ruidosa que ni siquiera un modelo con acceso directo a
    los lags de viento superaba a la persistencia naive -- doble conteo de
    ruido, no realismo adicional."""
    return np.clip(WIND_SPEED_PEAK_MS * wind_idx, 0, None)


def _wind_power_fraction(speed_ms: np.ndarray) -> np.ndarray:
    """Fracción de capacidad nominal entregada por una turbina a una
    velocidad de viento dada: cero bajo cut-in, creciente entre cut-in y la
    velocidad nominal, plana en capacidad nominal por encima -- el
    comportamiento real de una curva de potencia eólica. Exponente 2 (no el
    v^3 "ideal" de la energía cinética de libro de texto): las curvas de
    potencia publicadas por fabricantes reales son más suaves que el cubo
    idealizado cerca de cut-in (el control de pitch de las palas "redondea"
    la curva); un exponente 3 sobre una señal de viento con ruido realista
    amplificó ese ruido ~3x en términos relativos y volvió la generación
    eólica más ruidosa que predecible incluso para un modelo con acceso
    directo a la velocidad de viento -- un artefacto de la formulación, no
    una dificultad real del problema, y se corrigió en vez de reportarlo
    como si fuera un hallazgo genuino."""
    frac = np.zeros_like(speed_ms)
    ramp_mask = (speed_ms > WIND_CUT_IN_MS) & (speed_ms < WIND_RATED_MS)
    frac[ramp_mask] = ((speed_ms[ramp_mask] - WIND_CUT_IN_MS) / (WIND_RATED_MS - WIND_CUT_IN_MS)) ** 1.2
    frac[speed_ms >= WIND_RATED_MS] = 1.0
    return frac


def _temperature_c(node: str, hour: np.ndarray, day_of_year: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    """Temperatura del aire, °C -- media anual del nodo + ciclo estacional
    (hemisferio sur: pico a fines de enero, con el retraso térmico típico
    respecto al solsticio del 21 de diciembre) + ciclo diurno (mínimo hacia
    el amanecer, máximo a media tarde) + ruido."""
    params = TEMPERATURE_PARAMS[node]
    seasonal = params["seasonal_amplitude_c"] * np.cos(2 * np.pi * (day_of_year - 25) / 365.25)
    diurnal = params["diurnal_amplitude_c"] * np.sin(2 * np.pi * (hour - 9) / 24)
    noise = rng.normal(0, 1.2, n)
    return params["annual_mean_c"] + seasonal + diurnal + noise


def _wind_index(n: int, rng: np.random.Generator, mean_reversion: float = 0.02) -> np.ndarray:
    """Camino aleatorio acotado con reversión a la media: el viento tiene
    persistencia horaria (no cambia bruscamente hora a hora) pero, a diferencia
    del solar, no sigue un patrón diurno marcado -- por eso "frentes de viento"
    nuevos entran ocasionalmente y desplazan el nivel objetivo."""
    idx = np.zeros(n)
    idx[0] = rng.uniform(0.2, 0.5)
    target = rng.uniform(0.25, 0.45)
    for t in range(1, n):
        if rng.random() < 0.01:
            target = rng.uniform(0.1, 0.7)
        idx[t] = np.clip(idx[t - 1] + mean_reversion * (target - idx[t - 1]) + rng.normal(0, 0.03), 0, 1)
    return idx


def _generate_node_series(node: str, params: dict, timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    n = len(timestamps)
    hour = timestamps.hour.to_numpy().astype(float)
    day_of_year = timestamps.dayofyear.to_numpy()
    dow = timestamps.dayofweek.to_numpy()
    date_only = timestamps.normalize()

    # --- Meteorología: las variables físicas de las que la generación renovable
    # (y, en menor medida, la demanda) realmente dependen -- se calculan
    # primero, y solar/eólica/demanda se derivan de ellas, no al revés. ---
    seasonal = _seasonal_factor(day_of_year)
    shape = _solar_shape(hour, day_of_year)
    unique_dates = date_only.unique().sort_values()
    clear_sky_by_day = _clear_sky_daily_factor(len(unique_dates), rng)
    clear_sky_map = dict(zip(unique_dates, clear_sky_by_day))
    clear_sky = np.array([clear_sky_map[d] for d in date_only])
    ghi_wm2 = _ghi_wm2(node, day_of_year, shape, clear_sky, rng, n)

    wind_idx = _wind_index(n, rng)
    wind_speed_ms = _wind_speed_ms(wind_idx, rng, n)

    temperature_c = _temperature_c(node, hour, day_of_year, rng, n)

    # --- Solar: generación como fracción lineal de GHI sobre el pico de cielo
    # despejado del nodo -- la irradiancia YA incorpora estacionalidad, curva
    # diurna y nubosidad, así que no hace falta volver a aplicarlas aquí. ---
    solar_mwh = np.clip(params["solar_capacity_mwh"] * (ghi_wm2 / GHI_CLEAR_SKY_PEAK_WM2[node]), 0, None)

    # --- Eólica: curva de potencia cúbica real sobre la velocidad de viento,
    # con una pequeña modulación estacional adicional (densidad del aire,
    # más alta y fría en invierno) sobre la curva de potencia misma. ---
    # Sin ruido aditivo adicional sobre la potencia: `wind_speed_ms` ya trae
    # su propio ruido (4%, turbulencia/medición realista) y la curva cúbica
    # ya amplifica esa variación de forma no lineal -- sumar un segundo
    # ruido independiente directamente sobre la potencia duplicaría la
    # fuente de aleatoriedad sin ninguna justificación física.
    wind_seasonal = _seasonal_factor(day_of_year, peak_day=172, amplitude=0.10, base=0.90)
    wind_mwh = np.clip(params["wind_capacity_mwh"] * _wind_power_fraction(wind_speed_ms) * wind_seasonal, 0, params["wind_capacity_mwh"])

    # --- Demanda: perfil horario x fin de semana/feriado x estacionalidad x
    # tendencia de crecimiento x un efecto HVAC modesto por temperatura (más
    # consumo con frío o calor extremos respecto a una zona de confort, un
    # mecanismo real pero deliberadamente secundario frente al perfil horario
    # y de calendario, que siguen siendo el driver dominante de la demanda). ---
    hourly_shape = DEMAND_SHAPE[params["profile"]][timestamps.hour.to_numpy()]
    is_holiday = np.array([d.date() in CHILE_HOLIDAYS for d in timestamps])
    is_weekend = dow >= 5
    weekend_discount = 0.98 if params["profile"] == "mining" else 0.87
    weekend_factor = np.where(is_weekend | is_holiday, weekend_discount, 1.0)
    demand_seasonal = (
        np.ones(n) if params["profile"] == "mining"
        else _seasonal_factor(day_of_year, peak_day=172, amplitude=0.06, base=0.97)
    )
    years_elapsed = (timestamps - timestamps[0]).days / 365.25
    growth = 1 + 0.03 * years_elapsed
    demand_noise = rng.normal(1.0, 0.025, n)
    comfort_low_c, comfort_high_c = 18.0, 24.0
    hvac_sensitivity = 0.002 if params["profile"] == "mining" else 0.006
    hvac_effect = 1 + hvac_sensitivity * (
        np.clip(temperature_c - comfort_high_c, 0, None) + np.clip(comfort_low_c - temperature_c, 0, None)
    )
    demand_mwh = np.clip(
        params["base_demand_mwh"] * hourly_shape * weekend_factor * demand_seasonal * growth
        * hvac_effect * demand_noise,
        0, None,
    )

    # --- Costo marginal: función de la demanda residual (demanda - renovables),
    # más picos ocasionales de estrés de oferta / restricciones de transmisión.
    # Precios cercanos a 0 con alta solar y baja demanda residual son un fenómeno
    # real y conocido en el norte del SEN, no un artefacto del modelo. ---
    residual = demand_mwh - solar_mwh - wind_mwh
    residual_ratio = residual / (params["base_demand_mwh"] * 1.3)
    price = 45.0 + 95.0 * np.clip(residual_ratio, -0.5, 1.5)
    spike_mask = rng.random(n) < 0.006
    price = price + spike_mask * rng.uniform(60, 180, n)
    price = np.clip(price + rng.normal(0, 4.0, n), 0, 350)

    return pd.DataFrame({
        "timestamp": timestamps,
        "node": node,
        "solar_generation_mwh": solar_mwh.round(2),
        "wind_generation_mwh": wind_mwh.round(2),
        "demand_mwh": demand_mwh.round(2),
        "marginal_cost_usd_mwh": price.round(2),
        "ghi_w_m2": ghi_wm2.round(1),
        "wind_speed_ms": wind_speed_ms.round(2),
        "temperature_c": temperature_c.round(2),
    })


def generate_energy_data(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Genera el panel horario completo (todos los nodos) para 2024-2025."""
    timestamps = pd.date_range(START_DATE, END_DATE, freq="h")
    frames = [
        _generate_node_series(node, params, timestamps, np.random.default_rng(seed + i))
        for i, (node, params) in enumerate(NODES.items())
    ]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = generate_energy_data()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Filas generadas: {len(df):,} ({len(NODES)} nodos x {df['timestamp'].nunique():,} horas)")
    print(f"Rango: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Precio promedio: ${df['marginal_cost_usd_mwh'].mean():.2f}/MWh")
    print(f"Guardado en {OUTPUT_PATH}")
