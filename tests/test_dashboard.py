"""Smoke test del dashboard vía `streamlit.testing.v1.AppTest`: corre el script
completo (sin navegador) y falla si levanta cualquier excepción en el render.

Requiere que el pipeline ya se haya corrido (`fetch_energy_data` ->
`build_features` -> `train_forecaster`), igual que el dashboard en producción --
se salta automáticamente si los artefactos todavía no existen.
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_PATH = PROJECT_ROOT / "src" / "app" / "dashboard.py"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "forecaster_metrics.json"

pytestmark = pytest.mark.skipif(
    not METRICS_PATH.exists(),
    reason="Requiere haber corrido antes fetch_energy_data / build_features / train_forecaster.",
)


def test_dashboard_runs_without_exceptions():
    at = AppTest.from_file(str(DASHBOARD_PATH), default_timeout=120)
    at.run()
    assert not at.exception


def test_dashboard_renders_core_widgets():
    at = AppTest.from_file(str(DASHBOARD_PATH), default_timeout=120)
    at.run()
    assert len(at.metric) == 3
    assert len(at.dataframe) >= 1
