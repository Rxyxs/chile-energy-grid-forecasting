import numpy as np
import polars as pl
import pytest

from src.data.fetch_energy_data import generate_energy_data
from src.features.build_features import build_features
from src.models.conformal import (
    cqr_margin,
    evaluate_coverage,
    rectify_crossing,
    train_conformal_intervals,
    _split_train_calibration,
)

TINY_LGBM_PARAMS = {"n_estimators": 30, "learning_rate": 0.3, "max_depth": 3, "num_leaves": 7}


def test_split_train_calibration_partitions_without_overlap():
    train_ts = set(range(100))
    fit_ts, calib_ts = _split_train_calibration(train_ts, calibration_fraction=0.2)

    assert fit_ts | calib_ts == train_ts
    assert fit_ts & calib_ts == set()
    assert len(calib_ts) == 20


def test_split_train_calibration_keeps_calibration_chronologically_last():
    train_ts = set(range(100))
    fit_ts, calib_ts = _split_train_calibration(train_ts, calibration_fraction=0.2)

    assert max(fit_ts) < min(calib_ts)


def test_rectify_crossing_swaps_only_the_inverted_rows():
    """Encontrado corriendo forecast.py de verdad: dos cuantiles entrenados por
    separado pueden cruzarse fila a fila (aquí, la 2da y 4ta)."""
    lower = np.array([0.0, 5.0, -1.0, 0.17])
    upper = np.array([2.0, 1.0, 3.0, 0.03])

    fixed_lower, fixed_upper = rectify_crossing(lower, upper)

    assert list(fixed_lower) == [0.0, 1.0, -1.0, 0.03]
    assert list(fixed_upper) == [2.0, 5.0, 3.0, 0.17]
    assert (fixed_lower <= fixed_upper).all()


def test_evaluate_coverage_rectifies_crossed_intervals_before_scoring():
    y_true = np.array([1.5])
    lower = np.array([2.0])  # cruzado: lower > upper
    upper = np.array([1.0])

    result = evaluate_coverage(y_true, lower, upper)

    assert result["coverage"] == 1.0  # 1.5 cae dentro de [1.0, 2.0] una vez rectificado
    assert result["mean_width"] == pytest.approx(1.0)


def test_evaluate_coverage_counts_points_inside_the_interval():
    y_true = np.array([1.0, 5.0, 10.0, -3.0])
    lower = np.array([0.0, 0.0, 0.0, 0.0])
    upper = np.array([2.0, 6.0, 8.0, 1.0])

    result = evaluate_coverage(y_true, lower, upper)

    assert result["coverage"] == 0.5  # solo 1.0 y 5.0 caen dentro
    assert result["mean_width"] == pytest.approx(4.25)  # (2 + 6 + 8 + 1) / 4


def test_cqr_margin_matches_manual_finite_sample_quantile():
    rng = np.random.default_rng(0)
    y_calib = rng.normal(0, 1, size=200)
    lower_calib = np.full(200, -0.5)
    upper_calib = np.full(200, 0.5)

    margin = cqr_margin(y_calib, lower_calib, upper_calib, alpha=0.10)

    scores = np.maximum(lower_calib - y_calib, y_calib - upper_calib)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * 0.90) / n)
    expected = np.quantile(scores, level, method="higher")

    assert margin == pytest.approx(expected)


def test_conformal_correction_recovers_nominal_coverage_that_the_raw_quantiles_miss():
    """La ablación central de este módulo: si el par de cuantiles crudo es
    demasiado angosto (aquí, deliberadamente clavado en +-0.5 sobre una normal
    estándar, muy por debajo del +-1.645 que un intervalo 90% real necesita),
    la cobertura cruda debe quedar muy por debajo del 90% nominal -- y la
    corrección conformal, calculada sobre un tramo de calibración de la MISMA
    distribución, debe recuperar una cobertura cercana al nominal en un test
    set fresco. Si esto no se cumpliera, conformalizar no estaría haciendo lo
    que promete."""
    rng = np.random.default_rng(1)
    y_calib = rng.normal(0, 1, size=2_000)
    y_test = rng.normal(0, 1, size=5_000)

    narrow_lower, narrow_upper = -0.5, 0.5
    lower_calib = np.full_like(y_calib, narrow_lower)
    upper_calib = np.full_like(y_calib, narrow_upper)
    lower_test = np.full_like(y_test, narrow_lower)
    upper_test = np.full_like(y_test, narrow_upper)

    naive = evaluate_coverage(y_test, lower_test, upper_test)
    margin = cqr_margin(y_calib, lower_calib, upper_calib, alpha=0.10)
    conformal = evaluate_coverage(y_test, lower_test - margin, upper_test + margin)

    assert naive["coverage"] < 0.60  # el intervalo crudo, angosto a propósito, falla claramente
    assert conformal["coverage"] == pytest.approx(0.90, abs=0.03)


@pytest.mark.parametrize("target", ["marginal_cost_usd_mwh", "solar_generation_mwh"])
def test_train_conformal_intervals_runs_end_to_end_on_real_features(target):
    """Modelos reales (no mocks) pero minúsculos, igual que
    tests/test_forecast.py -- valida que el pipeline completo (split
    fit/calibración, entrenamiento de cuantiles, conformalización, evaluación)
    corre de punta a punta sobre el generador real, no la precisión del
    modelo."""
    raw = generate_energy_data()
    features = build_features(pl.from_pandas(raw)).to_pandas()

    result = train_conformal_intervals(features, target, n_splits=2, lgbm_params=TINY_LGBM_PARAMS)

    assert len(result["fold_metrics"]) == 2
    assert not np.isnan(result["margin"])
    for m in result["fold_metrics"]:
        assert 0.0 <= m["naive_coverage"] <= 1.0
        assert 0.0 <= m["conformal_coverage"] <= 1.0
        assert m["conformal_mean_width"] >= 0
