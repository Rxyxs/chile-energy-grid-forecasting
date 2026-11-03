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


def _hour_dependent_test(n_days: int = 240, seed: int = 0):
    """Ruido de sd 1 en todas las horas salvo las 12:00, con sd 1,3; la calibración
    tiene las mismas horas pero con sd 1 en todas, así que un nivel único subcubre
    las 12:00."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=24 * n_days, freq="h").to_numpy()
    hours = pd.DatetimeIndex(ts).hour.to_numpy()
    y = rng.normal(0, np.where(hours == 12, 1.3, 1.0))
    cal_hours = np.tile(np.arange(24), 60)
    cal = np.abs(rng.normal(0, 1.0, len(cal_hours)))
    zeros = np.zeros(len(y))
    return y, zeros, zeros, ts, cal, cal_hours, hours


def test_hourly_aci_covers_the_hour_a_single_level_misses():
    y, lo, hi, ts, cal, cal_hours, hours = _hour_dependent_test()
    noon = hours == 12
    late = np.arange(len(y)) >= len(y) // 2
    m_global, _ = S.online_intervals(y, lo, hi, ts, cal, "aci")
    m_hour = S.online_intervals_hourly(y, lo, hi, ts, cal, cal_hours)
    cov_global = (np.abs(y) <= m_global)[noon & late].mean()
    cov_hour = (np.abs(y) <= m_hour)[noon & late].mean()
    assert cov_hour > cov_global + 0.05
    assert cov_hour > 0.85


def test_hourly_aci_first_margin_comes_from_calibration_only():
    y, lo, hi, ts, cal, cal_hours, _ = _hour_dependent_test(n_days=2)
    y = y.copy()
    y[0] = 1e6
    m = S.online_intervals_hourly(y, lo, hi, ts, cal, cal_hours)
    assert m[0] == S.finite_quantile(cal[cal_hours == 0], S.ALPHA)
    assert m[24] > m[0]  # el fallo de la hora 0 amplía la hora 0 del día siguiente


def test_scaled_margin_is_proportional_to_the_raw_width():
    y, lo, hi, ts, cal, cal_hours, _ = _hour_dependent_test(n_days=1)
    width = np.linspace(1.0, 2.0, len(y))
    m = S.online_intervals_hourly(y, lo - width / 2, hi + width / 2, ts, cal, cal_hours,
                                  scale=width, calib_scale=np.ones(len(cal)))
    q = np.array([S.finite_quantile(cal[cal_hours == h], S.ALPHA) for h in range(24)])
    assert np.allclose(m, q * width)
