import math

import polars as pl

from src.features.build_features import (
    add_calendar_features,
    add_cyclical_features,
    add_holiday_feature,
    add_lag_features,
    add_rolling_features,
)


def _toy_panel() -> pl.DataFrame:
    """2 nodos x 48 horas, valores = índice de hora dentro del nodo (1..48),
    para poder calcular a mano lo que un lag/rolling correcto debería dar."""
    timestamps = pl.datetime_range(
        pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 3, 0), interval="1h", eager=True
    )[:-1]
    rows = []
    for node in ["A", "B"]:
        for i, ts in enumerate(timestamps):
            rows.append({
                "timestamp": ts, "node": node,
                "solar_generation_mwh": float(i + 1), "wind_generation_mwh": float(i + 1),
                "demand_mwh": float(i + 1), "marginal_cost_usd_mwh": float(i + 1),
            })
    return pl.DataFrame(rows)


def test_lag_features_use_previous_value_within_node():
    df = add_lag_features(_toy_panel())
    node_a = df.filter(pl.col("node") == "A").sort("timestamp")

    values = node_a["demand_mwh"].to_list()
    lag_1h = node_a["demand_mwh_lag_1h"].to_list()

    assert lag_1h[0] is None
    for i in range(1, len(values)):
        assert lag_1h[i] == values[i - 1]


def test_lag_features_never_cross_nodes():
    df = add_lag_features(_toy_panel())
    first_row_per_node = df.sort(["node", "timestamp"]).group_by("node", maintain_order=True).first()
    # La primera hora de cada nodo no tiene "hora anterior" -- ni siquiera la
    # última hora del otro nodo, que es lo que fallaría si el lag no fuera `.over("node")`.
    assert first_row_per_node["demand_mwh_lag_1h"].is_null().all()


def test_rolling_mean_excludes_current_row():
    df = add_rolling_features(_toy_panel())
    node_a = df.filter(pl.col("node") == "A").sort("timestamp")

    demand = node_a["demand_mwh"].to_list()
    rolling_mean_6h = node_a["demand_mwh_rolling_mean_6h"].to_list()

    # Fila 10 (0-indexed): la ventana de 6h debe promediar las filas 4..9
    # (shift(1) primero), NUNCA la fila 10 misma.
    idx = 10
    expected = sum(demand[idx - 6:idx]) / 6
    assert math.isclose(rolling_mean_6h[idx], expected, rel_tol=1e-9)


def test_cyclical_encoding_is_bounded_and_continuous_at_midnight():
    df = add_cyclical_features(_toy_panel())
    assert df["hour_sin"].min() >= -1.0001 and df["hour_sin"].max() <= 1.0001
    assert df["hour_cos"].min() >= -1.0001 and df["hour_cos"].max() <= 1.0001

    hour_23 = df.filter(pl.col("timestamp").dt.hour() == 23).select("hour_sin", "hour_cos").row(0)
    hour_0 = df.filter(pl.col("timestamp").dt.hour() == 0).select("hour_sin", "hour_cos").row(0)
    # 23h y 0h deben quedar cerca en el círculo, no en extremos opuestos como
    # quedarían con una codificación entera directa (23 vs 0).
    dist = math.dist(hour_23, hour_0)
    assert dist < 0.3


def test_calendar_features_flag_weekend():
    df = add_calendar_features(_toy_panel())
    # 2024-01-01 es lunes, 2024-01-02 es martes: ninguna hora debería marcar fin de semana.
    assert df["is_weekend"].sum() == 0


def test_holiday_feature_flags_new_years_day():
    df = add_holiday_feature(_toy_panel())
    jan_1 = df.filter(pl.col("timestamp").dt.date() == pl.date(2024, 1, 1))
    assert jan_1["is_holiday"].to_list() == [1] * jan_1.height
