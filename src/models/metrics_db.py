"""Persistencia de métricas comparativas (naive / seasonal-naive / LightGBM /
XGBoost / MLP PyTorch) en DuckDB local -- hasta ahora cada módulo escribía su
propio JSON (`forecaster_metrics.json`, `temporal_stability_metrics.json`,
`torch_forecaster_metrics.json`); este módulo los consolida en una única
tabla queryable (`model_comparison`) en `data/processed/metrics.duckdb`, sin
reemplazar los JSON existentes (siguen siendo la fuente detallada por fold).
"""
from __future__ import annotations

import time
from pathlib import Path

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "metrics.duckdb"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS model_comparison (
    target_column   VARCHAR,
    model_family    VARCHAR,   -- 'naive' | 'seasonal_naive' | 'lightgbm' | 'xgboost' | 'mlp_pytorch'
    model_variant    VARCHAR,  -- e.g. activation name for mlp_pytorch, else same as model_family
    mean_wape       DOUBLE,
    mean_mae        DOUBLE,
    n_splits        INTEGER,
    latency_ms_per_1k_rows DOUBLE,
    recorded_at     TIMESTAMP
)
"""


def get_connection(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(CREATE_TABLE_SQL)
    return con


def record_comparison_row(
    con: duckdb.DuckDBPyConnection,
    target_column: str,
    model_family: str,
    mean_wape: float,
    mean_mae: float,
    n_splits: int,
    model_variant: str | None = None,
    latency_ms_per_1k_rows: float | None = None,
) -> None:
    con.execute(
        "INSERT INTO model_comparison VALUES (?, ?, ?, ?, ?, ?, ?, current_timestamp)",
        [target_column, model_family, model_variant or model_family, mean_wape, mean_mae, n_splits, latency_ms_per_1k_rows],
    )


def measure_predict_latency_ms_per_1k_rows(predict_fn, X: pd.DataFrame) -> float:
    """Mide latencia de inferencia sobre `X` tal cual está (sin repetir filas
    artificialmente), normalizada a "ms por cada 1000 filas" para que sea
    comparable entre modelos evaluados sobre folds de distinto tamaño."""
    n_rows = max(len(X), 1)
    start = time.perf_counter()
    predict_fn(X)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms / n_rows * 1000


def read_comparison_table(db_path: Path = DB_PATH) -> pd.DataFrame:
    con = get_connection(db_path)
    try:
        return con.execute("SELECT * FROM model_comparison ORDER BY target_column, model_family, model_variant").fetchdf()
    finally:
        con.close()
