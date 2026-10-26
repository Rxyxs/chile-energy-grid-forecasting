"""Fuentes reales versionadas en data/real/ y el panel que se arma con ellas.

Las series horarias salen de gráficos de PDF, así que estos tests las contrastan
con fuentes independientes: la generación mensual oficial del SEN por
tecnología (datos.gob.cl) y el costo marginal mensual de Crucero publicado por
la CNE.
"""
import numpy as np
import pandas as pd
import pytest

from src.data import cen_daily_reports as C
from src.data import real_sources as R
from src.data.fetch_energy_data import generate_energy_data
from src.data.real_panel import (
    BAR_NODES, NODES, PRICE_TARGET, SYSTEM_NODE, SYSTEM_TARGETS, WEATHER_PATH,
    load_bar_cmg, load_system_hourly, rows_for_target,
)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return generate_energy_data()


@pytest.fixture(scope="module")
def system_monthly() -> pd.DataFrame:
    """Meses con informe en al menos 95% de los días, sumados por mes."""
    s = load_system_hourly()
    s["month"] = s["timestamp"].dt.strftime("%Y-%m")
    days = s.groupby("month")["timestamp"].apply(lambda t: t.dt.date.nunique())
    full = [m for m, d in days.items() if d >= 0.95 * pd.Period(m).days_in_month]
    monthly = s.groupby("month")[SYSTEM_TARGETS].sum().loc[full]
    return monthly.mul(pd.Series({m: pd.Period(m).days_in_month / days[m] for m in full}), axis=0)


def test_cached_weather_covers_every_bar_and_hour_without_gaps():
    w = pd.read_csv(WEATHER_PATH, parse_dates=["timestamp"])
    assert set(w["node"]) == set(BAR_NODES.values())
    assert w.groupby("node").size().eq(35_064).all()  # 2022-2025 hora a hora
    assert w[["ghi_w_m2", "wind_speed_ms", "temperature_c"]].notna().all().all()
    assert w["ghi_w_m2"].between(0, 1400).all()


def test_marginal_cost_reconstruction_matches_the_daily_averages_printed_in_each_report():
    """La recta que lleva cada gráfico a USD/MWh se ajusta con 6-8 barras y 2
    parámetros: el residuo contra los promedios impresos (redondeados a 0,1)
    mide la reconstrucción."""
    q = pd.read_csv(C.REPORT_QUALITY_PATH)
    ok = q[q["cmg_bars"] >= 6]
    assert len(ok) >= 0.95 * len(q)
    assert ok["cmg_max_resid_usd"].median() < 0.1
    assert (ok["cmg_max_resid_usd"] < 1.0).mean() > 0.99


def test_generation_shapes_need_one_scale_for_all_five_technologies():
    """Una sola escala (MW por punto) reproduce la energía diaria de las cinco
    tecnologías de la tabla del mismo informe."""
    q = pd.read_csv(C.REPORT_QUALITY_PATH)
    err = q["gen_max_rel_err"].dropna()
    assert len(err) >= 0.95 * len(q)
    assert err.median() < 0.01


def test_monthly_generation_matches_the_official_sen_series(system_monthly):
    """La generación total calza con la serie oficial dentro de 3% y la eólica
    dentro de 5%. La solar de los gráficos queda sistemáticamente bajo la
    oficial (2-8%): ambas fuentes no cuentan exactamente las mismas centrales."""
    g = R.load_generation().set_index("month").reindex(system_monthly.index)
    for col, real_col, tol in (("solar_generation_mwh", "solar_mwh", 0.10),
                               ("wind_generation_mwh", "wind_mwh", 0.05),
                               ("demand_mwh", "total_mwh", 0.03)):
        ratio = system_monthly[col] / g[real_col]
        assert ratio.between(1 - tol, 1 + tol).mean() > 0.9, (col, ratio.describe())


def test_crucero_monthly_marginal_cost_matches_the_cne_reports_until_mid_2024():
    """Hasta junio de 2024 el promedio mensual de las horas reconstruidas calza
    con el reporte mensual de la CNE. Desde julio de 2024 el informe diario del
    Coordinador da sistemáticamente menos que la CNE (9-30 USD/MWh); la
    reconstrucción sigue reproduciendo el valor diario que imprime el propio
    informe, así que la diferencia es entre las dos publicaciones (README §2)."""
    cmg = load_bar_cmg()
    cru = cmg[cmg["node"] == "Crucero_220kV"]
    by_month = cru.groupby(cru["timestamp"].dt.strftime("%Y-%m"))
    days = by_month["timestamp"].apply(lambda t: t.dt.date.nunique())
    full = [m for m, d in days.items() if d >= 0.95 * pd.Period(m).days_in_month]
    cne = R.load_cmg().set_index("month")
    common = [m for m in full if m in cne.index and m <= "2024-06"]
    assert len(common) >= 15
    diff = by_month[PRICE_TARGET].mean().loc[common] - cne.loc[common, "cmg_usd_mwh"]
    assert diff.abs().median() < 2.0


def test_panel_has_six_bars_plus_the_system_node(panel):
    assert set(panel["node"]) == set(NODES)
    assert panel.groupby("node").size().eq(35_064).all()
    # Las series del sistema son las mismas en las 7 filas de cada hora.
    spread = panel.groupby("timestamp")[SYSTEM_TARGETS].nunique(dropna=True)
    assert (spread <= 1).all().all()


def test_rows_for_target_never_counts_the_same_system_hour_seven_times(panel):
    for target in SYSTEM_TARGETS:
        assert set(rows_for_target(panel, target)["node"]) == {SYSTEM_NODE}
    assert SYSTEM_NODE not in set(rows_for_target(panel, PRICE_TARGET)["node"])


def test_system_node_price_is_the_mean_of_the_bars(panel):
    wide = panel.pivot(index="timestamp", columns="node", values=PRICE_TARGET).dropna()
    bars = wide[list(BAR_NODES.values())].mean(axis=1)
    np.testing.assert_allclose(wide[SYSTEM_NODE], bars, atol=0.01)


def test_simulator_mode_still_runs_without_any_real_source():
    df = generate_energy_data(use_real=False)
    assert len(df) == 87_720 and df.isna().sum().sum() == 0
