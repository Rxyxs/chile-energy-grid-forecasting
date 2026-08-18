"""Dashboard de monitoreo para operadores del SEN: precio real vs. pronosticado
por nodo, con alertas de precio pico configurables.

El pronóstico que se muestra es el del último fold de validación walk-forward de
`train_forecaster.py` -- predicciones genuinas sobre datos que el modelo nunca
vio en entrenamiento, no valores ajustados en muestra (que sobreestimarían la
precisión real y no servirían para juzgar si el modelo es confiable en producción).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import TimeSeriesSplit

from src.models.train_forecaster import (
    CATEGORICAL_FEATURES,
    DEFAULT_TARGET,
    N_SPLITS,
    get_feature_columns,
    load_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "forecaster_marginal_cost.joblib"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "forecaster_metrics.json"

st.set_page_config(page_title="Monitoreo SEN — Precios y Generación", page_icon="⚡", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_data() -> pd.DataFrame:
    return load_features()


@st.cache_data
def get_last_fold_predictions(_model, df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye el set de test del último fold de `TimeSeriesSplit` (idéntico
    al usado en `train_forecaster.py`) y genera predicciones fuera de muestra."""
    feature_columns = get_feature_columns(df)
    df = df.dropna(subset=feature_columns + [DEFAULT_TARGET]).reset_index(drop=True)

    unique_timestamps = np.sort(df["timestamp"].unique())
    splits = list(TimeSeriesSplit(n_splits=N_SPLITS).split(unique_timestamps))
    _, test_idx = splits[-1]
    test_ts = set(unique_timestamps[test_idx])

    test_df = df[df["timestamp"].isin(test_ts)].copy()
    X_test = test_df[feature_columns].copy()
    for col in CATEGORICAL_FEATURES:
        X_test[col] = X_test[col].astype("category")

    test_df["predicted"] = _model.predict(X_test)
    return test_df[[
        "timestamp", "node", DEFAULT_TARGET, "predicted",
        "solar_generation_mwh", "wind_generation_mwh", "demand_mwh",
    ]]


def render_alerts(view_df: pd.DataFrame, threshold: float, window_days: int, node: str) -> None:
    st.subheader("🚨 Alertas de precio pico")
    spikes = view_df[
        (view_df[DEFAULT_TARGET] >= threshold) | (view_df["predicted"] >= threshold)
    ].sort_values("timestamp", ascending=False)

    if spikes.empty:
        st.success(f"Sin eventos sobre ${threshold:.0f}/MWh en los últimos {window_days} días para {node}.")
        return

    st.warning(f"{len(spikes)} horas sobre ${threshold:.0f}/MWh en los últimos {window_days} días.")
    st.dataframe(
        spikes[[
            "timestamp", DEFAULT_TARGET, "predicted", "demand_mwh",
            "solar_generation_mwh", "wind_generation_mwh",
        ]].rename(columns={DEFAULT_TARGET: "precio_real_usd_mwh", "predicted": "precio_pronosticado_usd_mwh"}),
        use_container_width=True,
        height=min(400, 40 + 35 * len(spikes)),
    )


def render_price_chart(view_df: pd.DataFrame, threshold: float, node: str) -> None:
    st.subheader(f"Precio real vs. pronosticado — {node}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=view_df["timestamp"], y=view_df[DEFAULT_TARGET],
        name="Real", line=dict(color="#2a78d6", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=view_df["timestamp"], y=view_df["predicted"],
        name="Pronosticado", line=dict(color="#eb6834", width=2, dash="dot"),
    ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#e34948", annotation_text="Umbral de alerta")
    fig.update_layout(yaxis_title="$/MWh", xaxis_title="Fecha", height=420, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)


def render_generation_chart(view_df: pd.DataFrame, node: str) -> None:
    st.subheader(f"Generación renovable vs. demanda — {node}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view_df["timestamp"], y=view_df["demand_mwh"], name="Demanda", line=dict(color="#4a3aa7", width=2)))
    fig.add_trace(go.Scatter(x=view_df["timestamp"], y=view_df["solar_generation_mwh"], name="Solar", line=dict(color="#eda100", width=1.5)))
    fig.add_trace(go.Scatter(x=view_df["timestamp"], y=view_df["wind_generation_mwh"], name="Eólica", line=dict(color="#1baf7a", width=1.5)))
    fig.update_layout(yaxis_title="MWh", xaxis_title="Fecha", height=380, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.title("⚡ Monitoreo del Sistema Eléctrico Nacional (SEN)")
    st.caption(
        "Precio real vs. pronosticado y alertas de precio pico, sobre el set de "
        "evaluación fuera de muestra (walk-forward) del modelo entrenado."
    )

    model = load_model()
    metrics = load_metrics()
    df = load_data()
    predictions = get_last_fold_predictions(model, df)

    nodes = sorted(predictions["node"].unique())
    with st.sidebar:
        st.header("Controles")
        selected_node = st.selectbox("Nodo / Barra", nodes)
        alert_threshold = st.slider("Umbral de alerta de precio pico ($/MWh)", 50, 300, 150, step=5)
        window_days = st.slider("Ventana a mostrar (días)", 3, 60, 14)

    node_df = predictions[predictions["node"] == selected_node].sort_values("timestamp")
    cutoff = node_df["timestamp"].max() - pd.Timedelta(days=window_days)
    view_df = node_df[node_df["timestamp"] >= cutoff]

    m1, m2, m3 = st.columns(3)
    m1.metric("WAPE promedio (walk-forward)", f"{metrics['mean_wape']:.2f}%")
    m2.metric("MAE promedio", f"${metrics['mean_mae']:.2f}/MWh")
    m3.metric("Folds de validación", metrics["n_splits"])

    render_alerts(view_df, alert_threshold, window_days, selected_node)
    render_price_chart(view_df, alert_threshold, selected_node)
    render_generation_chart(view_df, selected_node)


if __name__ == "__main__":
    main()
