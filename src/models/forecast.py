"""Pronóstico recursivo multi-step-ahead (N horas) para uno o varios nodos,
usando los 4 modelos entrenados por `train_forecaster.py`.

Los 4 targets comparten el mismo set de features (`get_feature_columns`
siempre excluye los valores crudos de la hora actual de los 4, ver
`train_forecaster.py`), así que ninguno depende del valor *actual* de otro --
solo de sus lags/rolling ya calculados. Eso permite construir las features una
sola vez por hora futura y predecir los 4 targets con ese mismo vector, en vez
de tener que ordenar "qué se predice primero".

Cada hora futura se predice y se agrega de vuelta al buffer histórico antes de
avanzar a la siguiente -- así los lags/rolling de la hora t+2 ven la predicción
de t+1 como si fuera dato real, que es como funcionaría un pronóstico en
producción (no se conoce el futuro real todavía).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl

from src.features.build_features import (
    WEATHER_COLUMNS,
    add_calendar_features,
    add_cyclical_features,
    add_holiday_feature,
    add_lag_features,
    add_rolling_features,
    add_weather_lag_features,
)
from src.data.real_panel import PRICE_TARGET, SYSTEM_NODE, SYSTEM_TARGETS
from src.models.conformal import margin_for
from src.models.train_forecaster import CATEGORICAL_FEATURES, TARGET_COLUMNS, get_feature_columns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "sen_hourly_data.csv"
MODEL_DIR = PROJECT_ROOT / "data" / "processed"
FORECAST_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "forecast.csv"

# Suficiente para el lag más largo (168h) + la ventana rolling más larga (24h),
# con margen -- no hace falta cargar los 2 años completos para pronosticar
# unas pocas decenas de horas hacia adelante.
HISTORY_BUFFER_HOURS = 250

# Un pronóstico recursivo puede derivar fuera de rango tras muchos pasos, así
# que se recorta a lo físicamente posible (generación y costo marginal no
# negativos) antes de usarlo como "historia" del siguiente paso.
TARGET_CLIP_RANGES = {
    "solar_generation_mwh": (0.0, None),
    "wind_generation_mwh": (0.0, None),
    "demand_mwh": (0.0, None),
    "marginal_cost_usd_mwh": (0.0, None),
}


def load_models() -> dict[str, object]:
    models = {}
    for target in TARGET_COLUMNS:
        path = MODEL_DIR / f"forecaster_{target}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Falta {path} -- corre `python -m src.models.train_forecaster` primero.")
        models[target] = joblib.load(path)
    return models


def load_blend_weights() -> dict[str, float]:
    """Pesos de la mezcla modelo/persistencia del precio por barra (último fold de
    ``train_forecaster.walk_forward_blend``); vacío si no se entrenó la mezcla."""
    import json

    path = MODEL_DIR / "forecaster_metrics.json"
    if not path.exists():
        return {}
    metrics = json.loads(path.read_text(encoding="utf-8")).get(PRICE_TARGET, {})
    return metrics.get("blend_with_persistence", {}).get("deployed_weights", {})


def load_interval_models() -> dict[str, dict]:
    """Modelos de intervalo (cuantiles conformalizados, `src/models/conformal.py`)
    -- opcionales: si no se entrenaron todavía, `forecast_node` simplemente no
    agrega columnas de intervalo, en vez de fallar."""
    from src.models.conformal import load_conformal_artifacts

    return load_conformal_artifacts()


def _build_features_for_buffer(buffer: pl.DataFrame) -> pl.DataFrame:
    df = add_lag_features(buffer)
    df = add_weather_lag_features(df)
    df = add_rolling_features(df)
    df = add_cyclical_features(df)
    df = add_calendar_features(df)
    df = add_holiday_feature(df)
    return df


def forecast_node(
    node_history: pd.DataFrame,
    node: str,
    horizon: int,
    models: dict[str, object],
    interval_models: dict[str, dict] | None = None,
    system_path: pd.DataFrame | None = None,
    blend_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Pronostica `horizon` horas hacia adelante para un solo nodo, de forma
    recursiva. Si se pasa `interval_models` (ver `load_interval_models`),
    agrega columnas `<target>_lower`/`<target>_upper` por cada target que
    tenga modelo de intervalo -- calculadas sobre el mismo vector de features
    que el pronóstico puntual, en cada paso, pero sin retroalimentarse: lo
    único que avanza el buffer histórico de un paso al siguiente es la
    predicción puntual, igual que antes de agregar intervalos.

    ``system_path`` (pronóstico ya hecho del nodo SEN) aporta, en cada paso, la
    solar, eólica y demanda del sistema a la fila futura de una barra: el precio
    de la barra depende de esas series del sistema, que se pronostican una sola
    vez en el nodo SEN en vez de adivinarse desde el clima de cada barra.

    ``blend_weights`` (``load_blend_weights``) mezcla el precio pronosticado de la
    barra con su precio de la hora anterior, real o ya pronosticado."""
    buffer = node_history.tail(HISTORY_BUFFER_HOURS).reset_index(drop=True)
    feature_columns = None
    forecast_rows = []

    for _ in range(horizon):
        next_timestamp = buffer["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
        # El clima de la hora futura se persiste desde la última hora conocida
        # del buffer (no NaN, como los targets): esta simulación no incluye un
        # pronóstico meteorológico (NWP) real, y sin *algún* valor las
        # features de lag climático quedarían en NaN desde el segundo paso en
        # adelante, rompiendo el pronóstico recursivo. Persistencia es una
        # aproximación deliberadamente simple -- una limitación documentada,
        # no una que se intenta disimular -- de lo que en producción sería un
        # pronóstico NWP real para las horas futuras.
        last_known_weather = {col: buffer[col].iloc[-1] for col in WEATHER_COLUMNS}
        known = {}
        if system_path is not None:
            step = system_path[system_path["timestamp"] == next_timestamp]
            known = {t: float(step[t].iloc[0]) for t in SYSTEM_TARGETS if len(step) and t in step}
        new_row = pd.DataFrame([{
            "timestamp": next_timestamp, "node": node,
            **{t: known.get(t, np.nan) for t in TARGET_COLUMNS},
            **last_known_weather,
        }])
        extended = pd.concat([buffer, new_row], ignore_index=True)

        features_df = _build_features_for_buffer(pl.from_pandas(extended)).to_pandas()
        if feature_columns is None:
            feature_columns = get_feature_columns(features_df)

        X = features_df.iloc[[-1]][feature_columns].copy()
        for col in CATEGORICAL_FEATURES:
            X[col] = X[col].astype("category")

        predictions = {}
        for target, model in models.items():
            pred = float(model.predict(X)[0])
            lag = X.get(f"{target}_lag_1h")
            if target == PRICE_TARGET and blend_weights and node in blend_weights and lag is not None                     and pd.notna(lag.iloc[0]):
                w = blend_weights[node]
                pred = w * pred + (1 - w) * float(lag.iloc[0])
            low, high = TARGET_CLIP_RANGES[target]
            pred = max(pred, low) if low is not None else pred
            pred = min(pred, high) if high is not None else pred
            predictions[target] = pred

        intervals = {}
        for target, artifacts in (interval_models or {}).items():
            # Margen conformal de la hora del día de este paso (ver conformal.group_margins).
            margin = float(margin_for(artifacts["margin"], [next_timestamp.hour])[0])
            lower = float(artifacts["lower_model"].predict(X)[0]) - margin
            upper = float(artifacts["upper_model"].predict(X)[0]) + margin
            # Los dos cuantiles se entrenan por separado y pueden "cruzarse"
            # en una fila puntual (raro, pero ocurre en ~0.5% de las horas de
            # solar cerca de 0 MWh -- ver conformal.rectify_crossing).
            lower, upper = min(lower, upper), max(lower, upper)
            low, high = TARGET_CLIP_RANGES[target]
            lower = max(lower, low) if low is not None else lower
            upper = min(upper, high) if high is not None else upper
            intervals[f"{target}_lower"] = lower
            intervals[f"{target}_upper"] = upper

        new_row.loc[0, list(predictions.keys())] = list(predictions.values())
        buffer = pd.concat([buffer, new_row], ignore_index=True).tail(HISTORY_BUFFER_HOURS).reset_index(drop=True)
        forecast_rows.append({"timestamp": next_timestamp, "node": node, **known, **predictions, **intervals})

    return pd.DataFrame(forecast_rows)


def forecast(nodes: list[str], horizon: int, with_intervals: bool = True) -> pd.DataFrame:
    raw = pd.read_csv(RAW_PATH, parse_dates=["timestamp"])
    models = load_models()

    interval_models = None
    if with_intervals:
        try:
            interval_models = load_interval_models()
        except FileNotFoundError:
            interval_models = None  # opcional: corre `python -m src.models.conformal` para habilitarlos

    def history(node: str) -> pd.DataFrame:
        return raw[raw["node"] == node].sort_values("timestamp").reset_index(drop=True)

    def subset(targets: list[str], source: dict | None) -> dict | None:
        return None if source is None else {t: v for t, v in source.items() if t in targets}

    if SYSTEM_NODE not in set(raw["node"]):  # panel simulado: cada barra con sus 4 targets
        return pd.concat([forecast_node(history(n), n, horizon, models, interval_models) for n in nodes],
                         ignore_index=True)

    blend = load_blend_weights()
    # Panel real: primero el sistema (solar, eólica, demanda en el nodo SEN) y
    # después el precio de cada barra, con esa trayectoria del sistema.
    system_path = forecast_node(history(SYSTEM_NODE), SYSTEM_NODE, horizon,
                                subset(SYSTEM_TARGETS, models), subset(SYSTEM_TARGETS, interval_models))
    frames = [system_path] if SYSTEM_NODE in nodes else []
    for node in (n for n in nodes if n != SYSTEM_NODE):
        frames.append(forecast_node(history(node), node, horizon, subset([PRICE_TARGET], models),
                                    subset([PRICE_TARGET], interval_models), system_path=system_path,
                                    blend_weights=blend))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pronóstico recursivo multi-step-ahead para el SEN.")
    parser.add_argument("--horizon", type=int, default=24, help="Horas a pronosticar hacia adelante (default: 24).")
    parser.add_argument("--nodes", type=str, default="all", help="Nodos separados por coma, o 'all' (default).")
    parser.add_argument("--output", type=str, default=str(FORECAST_OUTPUT_PATH), help="Ruta del CSV de salida.")
    args = parser.parse_args()

    raw_df = pd.read_csv(RAW_PATH, usecols=["node"])
    all_nodes = sorted(raw_df["node"].unique())
    target_nodes = all_nodes if args.nodes == "all" else [n.strip() for n in args.nodes.split(",")]

    result = forecast(target_nodes, args.horizon)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Pronóstico de {args.horizon}h para {len(target_nodes)} nodo(s): {', '.join(target_nodes)}")
    print(result.to_string(index=False))
    print(f"\nGuardado en {output_path}")
