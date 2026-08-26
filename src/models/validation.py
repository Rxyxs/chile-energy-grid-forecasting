"""Utilidades de validación compartidas entre el entrenador LightGBM, los
baselines y el dashboard, para que todos midan sobre exactamente los mismos
folds y la misma métrica -- una comparación LightGBM-vs-baseline solo es
válida si ambos se evalúan sobre las mismas horas de test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (%), robusto a valores reales iguales a 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom * 100)


def get_walk_forward_folds(df: pd.DataFrame, n_splits: int) -> list[tuple[set, set]]:
    """Folds walk-forward sobre **timestamps únicos**, no sobre filas crudas del
    panel: con 5 nodos por hora, un split por fila podría dejar la hora 100 de
    un nodo en train y la hora 99 de otro en test -- eso sigue siendo lookahead
    bias aunque el índice de fila sea "posterior". Cada fold mueve todos los
    nodos de un mismo bloque temporal a la vez.
    """
    unique_timestamps = np.sort(df["timestamp"].unique())
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return [
        (set(unique_timestamps[train_idx]), set(unique_timestamps[test_idx]))
        for train_idx, test_idx in tscv.split(unique_timestamps)
    ]
