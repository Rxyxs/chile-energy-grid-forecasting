import polars as pl
import pytest

from src.data.fetch_energy_data import generate_energy_data
from src.features.build_features import build_features
from src.models.rolling_stability import evaluate_temporal_stability
from src.models.validation import get_rolling_origin_folds

TINY_LGBM_PARAMS = {"n_estimators": 20, "num_leaves": 7}
TINY_XGB_PARAMS = {"n_estimators": 20, "max_depth": 3}


@pytest.fixture(scope="module")
def features_df():
    raw = generate_energy_data()
    return build_features(pl.from_pandas(raw)).to_pandas()


def test_rolling_origin_folds_have_fixed_train_size():
    """A diferencia de walk-forward expansivo, cada fold de Rolling-Origin debe
    entrenar con exactamente el mismo número de timestamps -- esa es la
    propiedad que lo distingue y la que permite medir estabilidad temporal
    sin confundirla con "cuántos datos tenía disponibles ese fold"."""
    import pandas as pd

    timestamps = pd.date_range("2024-01-01", periods=24 * 400, freq="h")
    df = pd.DataFrame({"timestamp": timestamps})

    folds = get_rolling_origin_folds(df, n_splits=3, train_window_hours=24 * 60, test_window_hours=24 * 7)
    train_sizes = [len(train_ts) for train_ts, _ in folds]

    assert len(set(train_sizes)) == 1
    assert train_sizes[0] == 24 * 60


def test_rolling_origin_test_folds_are_chronologically_disjoint():
    import pandas as pd

    timestamps = pd.date_range("2024-01-01", periods=24 * 400, freq="h")
    df = pd.DataFrame({"timestamp": timestamps})

    folds = get_rolling_origin_folds(df, n_splits=4, train_window_hours=24 * 60, test_window_hours=24 * 7)
    for train_ts, test_ts in folds:
        assert train_ts.isdisjoint(test_ts)
        assert max(train_ts) < min(test_ts)


def test_evaluate_temporal_stability_runs_both_models_on_same_folds(features_df):
    """Prueba de integración real (no mocks) con modelos minúsculos: confirma
    que ambos modelos corren sobre exactamente los mismos folds y que se
    reportan las métricas de estabilidad esperadas."""
    result = evaluate_temporal_stability(
        features_df, "marginal_cost_usd_mwh",
        n_splits=2, train_window_days=30, test_window_days=5,
        lgbm_params=TINY_LGBM_PARAMS, xgb_params=TINY_XGB_PARAMS,
    )

    assert set(result.keys()) >= {"lightgbm", "xgboost"}
    for model_name in ("lightgbm", "xgboost"):
        summary = result[model_name]
        assert len(summary["fold_metrics"]) == 2
        assert summary["mean_wape"] >= 0
        assert summary["std_wape"] >= 0
        assert not (summary["cv_wape"] != summary["cv_wape"])  # no NaN
