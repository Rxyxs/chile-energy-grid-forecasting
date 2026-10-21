"""Fuentes reales versionadas en data/real/ y la calibración que ancla el panel a ellas."""
import numpy as np
import pandas as pd
import pytest

from src.data import real_sources as R
from src.data.fetch_energy_data import NODES, generate_energy_data


def test_cached_weather_covers_every_node_and_hour_without_gaps():
    w = R.load_weather()
    assert set(w["node"]) == set(NODES)
    assert w.groupby("node").size().eq(17_544).all()  # 2024-2025 hora a hora
    assert w[["ghi_w_m2", "wind_speed_ms", "temperature_c"]].notna().all().all()
    assert w["ghi_w_m2"].between(0, 1400).all()
    assert w["wind_speed_ms"].between(0, 60).all()


def test_monthly_generation_is_plausible_and_solar_peaks_in_summer():
    g = R.load_generation().set_index("month")
    sub = g.loc["2024-01":"2025-12"]
    assert len(sub) == 24
    assert sub["total_mwh"].between(5e6, 9e6).all()  # SEN: ~7 TWh/mes
    assert (sub["solar_mwh"] < sub["total_mwh"]).all()
    assert sub.loc["2024-12", "solar_mwh"] > sub.loc["2024-06", "solar_mwh"]  # hemisferio sur


def test_calibration_factors_move_the_monthly_shape_to_the_real_one():
    sim = pd.Series([1.0, 1.0, 1.0, 1.0], index=["a", "b", "c", "d"])
    real = pd.Series([2.0, 1.0, 1.0, 4.0], index=["a", "b", "c", "d"])
    f = R.calibration_factors(sim, real)
    assert R.monthly_shape(sim * f).to_numpy() == pytest.approx(R.monthly_shape(real).to_numpy())
    assert (sim * f).mean() == pytest.approx(sim.mean())  # el nivel promedio no cambia


def test_anchored_panel_matches_real_crucero_price_and_sen_monthly_shapes():
    df = generate_energy_data()
    month = df["timestamp"].dt.strftime("%Y-%m")
    crucero = df[df["node"] == "Crucero_220kV"].groupby(month)["marginal_cost_usd_mwh"].mean()
    cmg = R.load_cmg().set_index("month")["cmg_usd_mwh"].reindex(crucero.index)
    np.testing.assert_allclose(crucero.to_numpy(), cmg.to_numpy(), rtol=0.01)

    g = R.load_generation().set_index("month")
    for col, real_col in (("solar_generation_mwh", "solar_mwh"), ("demand_mwh", "total_mwh")):
        panel_shape = R.monthly_shape(df.groupby(month)[col].sum())
        real_shape = R.monthly_shape(g[real_col].reindex(panel_shape.index))
        np.testing.assert_allclose(panel_shape.to_numpy(), real_shape.to_numpy(), rtol=1e-3)


def test_simulator_mode_still_runs_without_any_real_source():
    df = generate_energy_data(use_real=False)
    assert len(df) == 87_720 and df.isna().sum().sum() == 0
