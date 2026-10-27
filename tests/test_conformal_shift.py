"""Calibración conformal bajo deriva: estacional, móvil y adaptativa (ACI)."""
import numpy as np
import pandas as pd

from src.models import conformal_shift as S


def _drifting_test(n_hours: int = 2000, seed: int = 0):
    """Intervalo crudo de ancho 0 alrededor de 0; el ruido real crece con el tiempo,
    así que un margen calibrado al comienzo queda corto al final."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n_hours, freq="h").to_numpy()
    sd = np.linspace(1.0, 3.0, n_hours)
    y = rng.normal(0, sd)
    calib_scores = np.abs(rng.normal(0, 1.0, 500))
    zeros = np.zeros(n_hours)
    return y, zeros, zeros, ts, calib_scores


def test_fixed_margin_undercovers_under_drift_and_online_methods_recover():
    y, lo, hi, ts, cal = _drifting_test()
    fixed = S.finite_quantile(cal, S.ALPHA)
    late = slice(1000, None)
    cov_fixed = (np.abs(y[late]) <= fixed).mean()
    for mode in ("rolling", "aci"):
        m, _ = S.online_intervals(y, lo, hi, ts, cal, mode)
        cov = (np.abs(y[late]) <= m[late]).mean()
        assert cov > cov_fixed + 0.1, mode


def test_online_margin_never_uses_the_hour_it_is_applied_to():
    """El margen de la primera hora sale solo de la calibración."""
    y, lo, hi, ts, cal = _drifting_test(n_hours=10)
    y = y.copy()
    y[0] = 1e6  # un valor extremo en la primera hora no puede ampliar su propio margen
    m, _ = S.online_intervals(y, lo, hi, ts, cal, "rolling")
    assert m[0] == S.finite_quantile(cal, S.ALPHA)
    assert m[1] >= m[0]


def test_seasonal_split_picks_the_same_calendar_weeks_of_earlier_years():
    train = set(pd.date_range("2022-01-01", "2023-12-31 23:00", freq="h"))
    test = set(pd.date_range("2024-07-01", "2024-07-14 23:00", freq="h"))
    fit, calib, used = S.seasonal_split(train, test, window_days=30)
    assert used and fit.isdisjoint(calib) and fit | calib == train
    months = pd.DatetimeIndex(sorted(calib)).month
    assert set(months) <= {5, 6, 7, 8}
    assert {2022, 2023} <= set(pd.DatetimeIndex(sorted(calib)).year)


def test_seasonal_split_falls_back_to_recent_when_there_is_no_earlier_season():
    train = set(pd.date_range("2024-01-01", "2024-02-29 23:00", freq="h"))
    test = set(pd.date_range("2024-08-01", "2024-08-07 23:00", freq="h"))
    _, calib, used = S.seasonal_split(train, test, window_days=20)
    assert not used and max(calib) == max(train)


def test_aci_widens_the_next_margin_after_a_run_of_misses():
    y, lo, hi, ts, cal = _drifting_test()
    _, next_margin = S.online_intervals(y, lo, hi, ts, cal, "aci")
    assert next_margin > S.finite_quantile(cal, S.ALPHA)
