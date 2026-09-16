"""Dashboard de monitoreo para operadores del SEN: precio real vs. pronosticado
por nodo, comparación contra baselines, y un pronóstico recursivo a futuro.

El pronóstico histórico que se muestra es el del último fold de validación
walk-forward de `train_forecaster.py` -- predicciones genuinas sobre datos que
el modelo nunca vio en entrenamiento, no valores ajustados en muestra (que
sobreestimarían la precisión real y no servirían para juzgar si el modelo es
confiable en producción). El panel de "pronóstico a futuro" usa el mismo
pronóstico recursivo multi-step de `forecast.py`, sobre horas que ni siquiera
existen en el dataset histórico.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.models.forecast import RAW_PATH, forecast_node, load_interval_models, load_models
from src.models.train_forecaster import (
    CATEGORICAL_FEATURES,
    DEFAULT_TARGET,
    get_feature_columns,
    load_features,
)
from src.models.validation import get_walk_forward_folds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "forecaster_metrics.json"

st.set_page_config(page_title="Monitoreo SEN — Precios y Generación", page_icon="⚡", layout="wide")


@st.cache_resource
def load_price_model():
    return load_models()[DEFAULT_TARGET]


@st.cache_resource
def load_all_models():
    return load_models()


@st.cache_resource
def load_all_interval_models():
    """Modelos de intervalo conformalizado (opcional): si `conformal.py`
    todavía no corrió, el panel de pronóstico a futuro simplemente no dibuja
    la banda, en vez de romper el dashboard entero."""
    try:
        return load_interval_models()
    except FileNotFoundError:
        return None


@st.cache_data
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_data() -> pd.DataFrame:
    return load_features()


@st.cache_data
def load_raw_history() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH, parse_dates=["timestamp"])


@st.cache_data
def get_last_fold_predictions(_model, df: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    """Reconstruye el set de test del último fold walk-forward (mismos folds que
    `train_forecaster.py`, vía `validation.get_walk_forward_folds`) y genera
    predicciones fuera de muestra."""
    feature_columns = get_feature_columns(df)
    df = df.dropna(subset=feature_columns + [DEFAULT_TARGET]).reset_index(drop=True)

    _, test_ts = get_walk_forward_folds(df, n_splits)[-1]
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
    st.subheader(f"Precio real vs. pronosticado (fuera de muestra) — {node}")
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


def render_baseline_comparison(metrics: dict) -> None:
    st.subheader("📊 LightGBM vs. baselines (mismos folds, todos los targets)")
    rows = []
    for target, m in metrics.items():
        rows.append({
            "target": target,
            "WAPE LightGBM": m["lightgbm"]["mean_wape"],
            "WAPE naive (t-1h)": m["naive_baseline"]["mean_wape"],
            "WAPE estacional (t-24h)": m["seasonal_naive_baseline"]["mean_wape"],
        })
    st.dataframe(pd.DataFrame(rows).round(2), use_container_width=True, hide_index=True)


def render_future_forecast(node: str, node_history: pd.DataFrame) -> None:
    st.subheader(f"🔮 Pronóstico a futuro (recursivo, multi-step) — {node}")
    horizon = st.slider("Horizonte a pronosticar (horas)", 6, 72, 24, step=6, key="horizon")

    models = load_all_models()
    interval_models = load_all_interval_models()
    future_df = forecast_node(node_history, node, horizon, models, interval_models)

    history_tail = node_history.tail(72)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history_tail["timestamp"], y=history_tail[DEFAULT_TARGET],
        name="Histórico", line=dict(color="#2a78d6", width=2),
    ))

    lower_col, upper_col = f"{DEFAULT_TARGET}_lower", f"{DEFAULT_TARGET}_upper"
    has_interval = interval_models is not None and lower_col in future_df.columns
    if has_interval:
        fig.add_trace(go.Scatter(
            x=pd.concat([future_df["timestamp"], future_df["timestamp"][::-1]]),
            y=pd.concat([future_df[upper_col], future_df[lower_col][::-1]]),
            fill="toself", fillcolor="rgba(27,175,122,0.15)", line=dict(color="rgba(0,0,0,0)"),
            name="Intervalo 90% (conformal)", hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=future_df["timestamp"], y=future_df[DEFAULT_TARGET],
        name=f"Pronóstico +{horizon}h", line=dict(color="#1baf7a", width=2, dash="dot"),
    ))
    fig.update_layout(yaxis_title="$/MWh", xaxis_title="Fecha", height=380, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Cada hora futura se predice usando las horas previas (reales o ya "
        "pronosticadas) como historia -- así es como funcionaría en producción, "
        "donde el futuro real todavía no existe."
        + (
            " La banda sombreada es el intervalo de predicción al 90% "
            "(Conformalized Quantile Regression, `src/models/conformal.py`), "
            "calibrado con cobertura empírica medida, no solo prometida."
            if has_interval
            else " Corre `python -m src.models.conformal` para habilitar la banda de incertidumbre."
        )
    )


def main() -> None:
    st.title("⚡ Monitoreo del Sistema Eléctrico Nacional (SEN)")
    st.caption(
        "Precio real vs. pronosticado, comparación contra baselines, y pronóstico "
        "recursivo a futuro, sobre un modelo LightGBM validado walk-forward."
    )

    metrics = load_metrics()
    price_metrics = metrics[DEFAULT_TARGET]["lightgbm"]

    model = load_price_model()
    df = load_data()
    predictions = get_last_fold_predictions(model, df, price_metrics["n_splits"])
    raw_history = load_raw_history()

    nodes = sorted(predictions["node"].unique())
    with st.sidebar:
        st.header("Controles")
        selected_node = st.selectbox("Nodo / Barra", nodes)
        alert_threshold = st.slider("Umbral de alerta de precio pico ($/MWh)", 50, 300, 150, step=5)
        window_days = st.slider("Ventana histórica a mostrar (días)", 3, 60, 14)

    node_df = predictions[predictions["node"] == selected_node].sort_values("timestamp")
    cutoff = node_df["timestamp"].max() - pd.Timedelta(days=window_days)
    view_df = node_df[node_df["timestamp"] >= cutoff]

    m1, m2, m3 = st.columns(3)
    m1.metric("WAPE promedio (walk-forward)", f"{price_metrics['mean_wape']:.2f}%")
    m2.metric("MAE promedio", f"${price_metrics['mean_mae']:.2f}/MWh")
    m3.metric("Folds de validación", price_metrics["n_splits"])

    render_alerts(view_df, alert_threshold, window_days, selected_node)
    render_price_chart(view_df, alert_threshold, selected_node)
    render_generation_chart(view_df, selected_node)
    render_baseline_comparison(metrics)

    node_history = raw_history[raw_history["node"] == selected_node].sort_values("timestamp")
    render_future_forecast(selected_node, node_history)


if __name__ == "__main__":
    main()
