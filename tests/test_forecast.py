import pandas as pd
import polars as pl
import pytest

from src.data.fetch_energy_data import NODES, generate_energy_data
from src.features.build_features import build_features
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
