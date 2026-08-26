"""Baselines de referencia (persistencia y estacional) evaluados sobre
exactamente los mismos folds walk-forward que el LightGBM, para que la
comparación en el README sea honesta -- un modelo aprendido que no vence a
"repetir el valor de hace 24h" no está aportando nada real.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.validation import get_walk_forward_folds, wape

NAIVE_LAG_SUFFIX = "_lag_1h"
SEASONAL_LAG_SUFFIX = "_lag_24h"


def _lag_column_metrics(df: pd.DataFrame, target_column: str, lag_column: str, n_splits: int) -> dict:
    subset = df.dropna(subset=[target_column, lag_column]).reset_index(drop=True)

    fold_metrics = []
    for fold, (train_ts, test_ts) in enumerate(get_walk_forward_folds(subset, n_splits)):
        test_df = subset[subset["timestamp"].isin(test_ts)]
        y_true = test_df[target_column].to_numpy()
        y_pred = test_df[lag_column].to_numpy()
        fold_metrics.append({
            "fold": fold,
            "test_rows": int(len(test_df)),
            "wape": wape(y_true, y_pred),
            "mae": float(np.abs(y_true - y_pred).mean()),
        })

    return {
        "method": lag_column,
        "n_splits": n_splits,
        "fold_metrics": fold_metrics,
        "mean_wape": float(np.mean([m["wape"] for m in fold_metrics])),
        "mean_mae": float(np.mean([m["mae"] for m in fold_metrics])),
    }


def naive_metrics(df: pd.DataFrame, target_column: str, n_splits: int) -> dict:
    """Persistencia: pronostica el valor de la hora anterior (`<target>_lag_1h`)."""
    return _lag_column_metrics(df, target_column, f"{target_column}{NAIVE_LAG_SUFFIX}", n_splits)


def seasonal_naive_metrics(df: pd.DataFrame, target_column: str, n_splits: int) -> dict:
    """Estacional: pronostica el valor de la misma hora, un día atrás (`<target>_lag_24h`) --
    captura el patrón diurno sin necesitar ningún modelo entrenado."""
    return _lag_column_metrics(df, target_column, f"{target_column}{SEASONAL_LAG_SUFFIX}", n_splits)
