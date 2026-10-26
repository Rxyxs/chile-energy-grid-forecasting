"""Piezas del extractor de los informes diarios del Coordinador, sin PDFs."""
from datetime import date

import numpy as np
import pytest

from src.data import cen_daily_reports as C


def test_report_urls_try_this_month_and_next_with_version_suffixes():
    urls = C.report_urls(date(2025, 1, 30))
    assert any(u.endswith("/2025/01/Resumen-Ejecutivo-de-Operacion-30-01-2025.pdf") for u in urls)
    assert any(u.endswith("/2025/02/Resumen-Ejecutivo-de-Operacion-30-01-2025-V1.pdf") for u in urls)
    # A mitad de mes no hace falta probar la carpeta del mes siguiente.
    assert all("/2025/01/" in u for u in C.report_urls(date(2025, 1, 10)))


@pytest.mark.parametrize("text, value", [("105.51", 105.51), ("1.126,57", 1126.57), ("37,91", 37.91), ("0.00", 0.0)])
def test_parse_number_cl_reads_both_report_formats(text, value):
    assert C.parse_number_cl(text) == pytest.approx(value)


def _row(y, *tokens):
    """Palabras de una fila como las entrega pymupdf: (x0, y0, x1, y1, texto)."""
    return [(10.0 * i, y, 10.0 * i + 8, y + 7, t) for i, t in enumerate(tokens)]


def test_parse_daily_table_keeps_the_gwh_row_and_skips_axis_and_mwh_rows():
    words = (
        _row(100, "Solar", "1", "4", "7", "10")                       # eje del gráfico: sin %
        + _row(200, "Solar", "44.65", "17.95%", "44.65", "44.23")     # tabla del día (GWh)
        + _row(300, "Solar", "44653.57", "47461.50", "-5.93%")        # tiempo real, en MWh
        + _row(400, "Térmica", "122.60", "49.28%")
        + _row(500, "Eólica", "40,82", "17,24", "%", "179,35")       # formato nuevo: % separado
    )
    table = C.parse_daily_table(words)
    assert table == {"solar": 44.65, "termica": 122.60, "eolica": 40.82}


def test_vertical_extent_measures_a_stacked_band_without_vertices_at_every_hour():
    # Banda entre y=10 (arriba) e y=30 (abajo) de x=0 a x=10; el borde superior
    # solo tiene sus extremos, como un tramo plano al que el PDF le quitó los
    # puntos colineales.
    band = np.array([(0, 10), (10, 10), (10, 30), (5, 30), (0, 30)], dtype=float)
    assert C.vertical_extent(band, 3.0) == pytest.approx(20.0)
    assert C.vertical_extent(band, 10.0) == pytest.approx(20.0)
    assert C.vertical_extent(band, 20.0) == 0.0


def test_vertical_extent_follows_a_sloped_top_edge():
    tri = np.array([(0, 30), (10, 10), (10, 30)], dtype=float)  # sube de 0 a 20 de alto
    assert C.vertical_extent(tri, 5.0) == pytest.approx(10.0)
    assert C.vertical_span(tri, 5.0) == pytest.approx((20.0, 30.0))


def test_bar_names_cover_both_legend_formats():
    assert C._bar_name("Pan de Azúcar") == C._bar_name("P.AZUCAR") == "Pan de Azucar"
    assert C._bar_name("Puerto Montt") == C._bar_name("P.MONTT") == "Puerto Montt"
    assert C._bar_name("Alto Jahuel") is None


def test_to_hourly_handles_quarter_hours_and_daylight_saving_days():
    assert C.to_hourly(np.repeat(np.arange(24.0), 4)).tolist() == list(range(24))
    spring = C.to_hourly(np.arange(1.0, 24.0))          # 23 horas: falta la primera
    assert np.isnan(spring[0]) and spring[1:].tolist() == list(range(1, 24))
    autumn = C.to_hourly(np.r_[np.arange(23.0), 10.0, 20.0])  # 25 horas: la repetida se promedia
    assert len(autumn) == 24 and autumn[-1] == 15.0
    assert len(C.to_hourly(np.zeros(92))) == 24


def test_parse_daily_table_accepts_rows_whose_percent_sign_is_missing():
    words = _row(100, "Solar", "43,41", "17,82", "134,75") + _row(200, "Solar", "1", "4", "7")
    assert C.parse_daily_table(words) == {"solar": 43.41}
