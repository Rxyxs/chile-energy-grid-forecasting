import pandas as pd

from src.features.build_features import WEATHER_COLUMNS
from src.models.train_forecaster import ID_COLUMNS, TARGET_COLUMNS, get_feature_columns


def _toy_features_df() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
        "node": ["A"] * 5,
        "solar_generation_mwh": range(5),
        "wind_generation_mwh": range(5),
        "demand_mwh": range(5),
        "marginal_cost_usd_mwh": range(5),
        "ghi_w_m2": range(5),
        "wind_speed_ms": range(5),
        "temperature_c": range(5),
        "ghi_w_m2_lag_1h": range(5),
        "wind_speed_ms_lag_24h": range(5),
        "solar_generation_mwh_lag_24h": range(5),
        "hour_sin": range(5),
    })


def test_get_feature_columns_excludes_raw_targets_and_ids():
    columns = get_feature_columns(_toy_features_df())
    for excluded in [*ID_COLUMNS, *TARGET_COLUMNS]:
        assert excluded not in columns


def test_get_feature_columns_excludes_raw_weather_but_keeps_weather_lags():
    columns = get_feature_columns(_toy_features_df())

    for raw_weather_col in WEATHER_COLUMNS:
        assert raw_weather_col not in columns

    assert "ghi_w_m2_lag_1h" in columns
    assert "wind_speed_ms_lag_24h" in columns


def test_get_feature_columns_keeps_node_and_engineered_features():
    columns = get_feature_columns(_toy_features_df())
    assert "node" in columns
    assert "hour_sin" in columns
    assert "solar_generation_mwh_lag_24h" in columns
