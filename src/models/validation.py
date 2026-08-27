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
    """Folds walk-forward **expansivos** sobre **timestamps únicos**, no sobre
    filas crudas del panel: con 5 nodos por hora, un split por fila podría
    dejar la hora 100 de un nodo en train y la hora 99 de otro en test -- eso
    sigue siendo lookahead bias aunque el índice de fila sea "posterior".
    Cada fold mueve todos los nodos de un mismo bloque temporal a la vez.

    "Expansivo": el set de train de cada fold sucesivo *acumula* todo el
    historial anterior (nunca se descarta), así que el último fold entrena
    con casi 2 años de datos y el primero con solo unos meses -- bueno para
    maximizar los datos de entrenamiento del modelo final, pero no aísla si
    el error de un fold viene del propio modelo o de cuánto historial tenía
    disponible. Para eso, ver `get_rolling_origin_folds`.
    """
    unique_timestamps = np.sort(df["timestamp"].unique())
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return [
        (set(unique_timestamps[train_idx]), set(unique_timestamps[test_idx]))
        for train_idx, test_idx in tscv.split(unique_timestamps)
    ]


def get_rolling_origin_folds(
    df: pd.DataFrame, n_splits: int, train_window_hours: int, test_window_hours: int
) -> list[tuple[set, set]]:
    """Folds walk-forward de **ventana deslizante fija** (Rolling-Origin CV):
    a diferencia de `get_walk_forward_folds` (que acumula todo el historial),
    cada fold entrena con exactamente `train_window_hours` horas más
    recientes -- ni más ni menos -- y valida sobre las `test_window_hours`
    horas siguientes, deslizando el origen hacia adelante en cada fold.

    El objetivo no es maximizar datos de entrenamiento (para eso está
    `get_walk_forward_folds`, usado para el modelo final) sino **aislar la
    estabilidad temporal del modelo**: si el WAPE de un modelo entrenado
    siempre con la misma cantidad de historial reciente varía mucho de un
    bloque de tiempo a otro, eso es evidencia de que el modelo (o el
    fenómeno que pronostica) no es temporalmente estable -- no un artefacto
    de que un fold tardío simplemente tuvo más datos para entrenar que uno
    temprano, que es exactamente la confusión que introduciría medir
    estabilidad sobre folds expansivos.

    Implementado sobre `TimeSeriesSplit(max_train_size=...)` de scikit-learn,
    que trunca cada train set al final de la ventana en vez de acumularlo --
    la primitiva nativa correcta para esto, no una reimplementación paralela.
    """
    unique_timestamps = np.sort(df["timestamp"].unique())
    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=train_window_hours, test_size=test_window_hours)
    return [
        (set(unique_timestamps[train_idx]), set(unique_timestamps[test_idx]))
        for train_idx, test_idx in tscv.split(unique_timestamps)
    ]
