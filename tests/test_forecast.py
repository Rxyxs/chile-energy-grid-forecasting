import pandas as pd
import polars as pl
import pytest

from src.data.fetch_energy_data import NODES, generate_energy_data
from src.features.build_features import build_features
from src.models.conformal import train_conformal_intervals
from src.models.forecast import forecast_node
from src.models.train_forecaster import TARGET_COLUMNS, train_with_walk_forward_validation

TINY_LGBM_PARAMS = {"n_estimators": 20, "learning_rate": 0.3, "max_depth": 3, "num_leaves": 7}
TEST_NODE = next(iter(NODES))


@pytest.fixture(scope="module")
def tiny_models() -> dict:
    """Modelos reales (no mocks) pero minúsculos y entrenados sobre pocos folds
    -- este test valida que el pronóstico recursivo multi-step corre de punta a
    punta y produce salidas físicamente válidas, no la precisión del modelo."""
    raw = generate_energy_data()
    features = build_features(pl.from_pandas(raw)).to_pandas()

    return {
        target: train_with_walk_forward_validation(
            features, target_column=target, n_splits=2, lgbm_params=TINY_LGBM_PARAMS
        )["final_model"]
        for target in TARGET_COLUMNS
    }


@pytest.fixture(scope="module")
def node_history() -> pd.DataFrame:
    raw = generate_energy_data()
    return raw[raw["node"] == TEST_NODE].sort_values("timestamp").reset_index(drop=True)


@pytest.fixture(scope="module")
def tiny_interval_models() -> dict:
    """Modelos de intervalo reales pero minúsculos (mismo patrón que
    `tiny_models`) -- valida que `forecast_node` agrega columnas de intervalo
    lógicamente válidas de punta a punta, no la cobertura empírica del CQR
    (eso ya lo valida `tests/test_conformal.py` con más folds y más datos)."""
    raw = generate_energy_data()
    features = build_features(pl.from_pandas(raw)).to_pandas()

    interval_models = {}
    for target in TARGET_COLUMNS:
        result = train_conformal_intervals(features, target, n_splits=2, lgbm_params=TINY_LGBM_PARAMS)
        interval_models[target] = {
            "lower_model": result["lower_model"], "upper_model": result["upper_model"], "margin": result["margin"],
        }
    return interval_models


def test_forecast_returns_one_row_per_future_hour(tiny_models, node_history):
    horizon = 5
    result = forecast_node(node_history, TEST_NODE, horizon, tiny_models)
    assert len(result) == horizon


def test_forecast_timestamps_are_hourly_and_continue_from_history(tiny_models, node_history):
    result = forecast_node(node_history, TEST_NODE, 5, tiny_models)
    last_history_ts = node_history["timestamp"].iloc[-1]

    assert result["timestamp"].iloc[0] == last_history_ts + pd.Timedelta(hours=1)
    diffs = result["timestamp"].diff().dropna()
    assert (diffs == pd.Timedelta(hours=1)).all()


def test_forecast_has_no_missing_targets(tiny_models, node_history):
    result = forecast_node(node_history, TEST_NODE, 5, tiny_models)
    assert result[TARGET_COLUMNS].isna().sum().sum() == 0


def test_forecast_respects_physical_bounds(tiny_models, node_history):
    result = forecast_node(node_history, TEST_NODE, 5, tiny_models)
    assert (result["solar_generation_mwh"] >= 0).all()
    assert (result["wind_generation_mwh"] >= 0).all()
    assert (result["demand_mwh"] >= 0).all()
    assert result["marginal_cost_usd_mwh"].between(0, 350).all()


def test_forecast_without_interval_models_has_no_interval_columns(tiny_models, node_history):
    result = forecast_node(node_history, TEST_NODE, 5, tiny_models)
    assert not any(col.endswith(("_lower", "_upper")) for col in result.columns)


def test_forecast_intervals_are_never_crossed(tiny_models, tiny_interval_models, node_history):
    """Regresión directa de un defecto real: al correr `forecast.py` de punta a
    punta, `solar_generation_mwh_lower` salió mayor que `..._upper` en una hora
    de transición día/noche, porque los dos cuantiles se entrenan por separado
    y nada les impide cruzarse (ver `conformal.rectify_crossing`)."""
    result = forecast_node(node_history, TEST_NODE, 8, tiny_models, tiny_interval_models)
    for target in TARGET_COLUMNS:
        assert (result[f"{target}_lower"] <= result[f"{target}_upper"]).all()
