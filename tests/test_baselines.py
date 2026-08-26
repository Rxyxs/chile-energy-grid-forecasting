import pandas as pd
import pytest

from src.models.baselines import naive_metrics, seasonal_naive_metrics


def _panel_with_known_lags() -> pd.DataFrame:
    """1 nodo, 100 horas. target = índice de hora; lag_1h/lag_24h se construyen
    a mano (no vía build_features) para que el WAPE esperado sea calculable."""
    n = 100
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h")
    target = pd.Series(range(1, n + 1), dtype=float)

    return pd.DataFrame({
        "timestamp": timestamps,
        "node": "A",
        "marginal_cost_usd_mwh": target,
        "marginal_cost_usd_mwh_lag_1h": target.shift(1),
        "marginal_cost_usd_mwh_lag_24h": target.shift(24),
    })


def test_naive_metrics_perfect_on_linear_trend():
    """Con un target que sube exactamente 1 unidad por hora, la persistencia
    (lag_1h) se equivoca por exactamente 1 en cada punto -- un WAPE positivo
    pero acotado, no cero (no es una predicción perfecta) y no NaN."""
    df = _panel_with_known_lags()
    result = naive_metrics(df, "marginal_cost_usd_mwh", n_splits=3)

    assert result["method"] == "marginal_cost_usd_mwh_lag_1h"
    assert result["mean_wape"] > 0
    assert not pd.isna(result["mean_wape"])


def test_seasonal_naive_uses_lag_24h_column():
    df = _panel_with_known_lags()
    result = seasonal_naive_metrics(df, "marginal_cost_usd_mwh", n_splits=3)
    assert result["method"] == "marginal_cost_usd_mwh_lag_24h"


def test_naive_beats_seasonal_on_a_pure_linear_trend():
    """En una tendencia lineal sin estacionalidad diaria, repetir la hora
    anterior (lag_1h, error=1 siempre) es estrictamente mejor que repetir la
    misma hora de ayer (lag_24h, error=24 siempre) -- si esto no se cumpliera,
    alguno de los dos baselines estaría mal calculado."""
    df = _panel_with_known_lags()
    naive = naive_metrics(df, "marginal_cost_usd_mwh", n_splits=3)
    seasonal = seasonal_naive_metrics(df, "marginal_cost_usd_mwh", n_splits=3)
    assert naive["mean_wape"] < seasonal["mean_wape"]


def test_metrics_have_one_entry_per_fold():
    df = _panel_with_known_lags()
    result = naive_metrics(df, "marginal_cost_usd_mwh", n_splits=3)
    assert len(result["fold_metrics"]) == 3
