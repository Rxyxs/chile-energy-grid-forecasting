"""Mezcla por barra entre el modelo de precio y la persistencia (el precio de hace 1 hora).

El precio de algunas barras es tan persistente, y tan poco ligado al estado
del sistema, que un modelo compartido entre barras reacciona de más: en Puerto
Montt LightGBM solo pierde contra la persistencia. Mezclar
``w * modelo + (1 - w) * precio_1h_antes`` con un peso por barra corrige eso sin
quitarle al modelo lo que sí sabe en el norte.

El peso de cada barra se elige en el último 15% del entrenamiento de cada fold,
con un modelo ajustado solo en el 85% anterior, y recién después se ajusta el
modelo con todo el entrenamiento y se aplica al test: el test nunca participa
en la elección del peso.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

BLEND_GRID = np.linspace(0, 1, 21)
VALIDATION_FRACTION = 0.15


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], target: str, params: dict,
                categorical: list[str]) -> np.ndarray:
    X_tr, X_te = train[cols].copy(), test[cols].copy()
    cats = [c for c in categorical if c in cols]
    for c in cats:
        X_tr[c] = X_tr[c].astype("category")
        X_te[c] = X_te[c].astype("category")
    model = LGBMRegressor(**params)
    model.fit(X_tr, train[target], categorical_feature=cats or "auto")
    return model.predict(X_te)


def blend_weights(train: pd.DataFrame, cols: list[str], target: str, params: dict, categorical: list[str],
                  lag_col: str, validation_fraction: float = VALIDATION_FRACTION) -> dict[str, float]:
    """Peso del modelo (vs. la persistencia) por nodo, elegido en el tramo más
    reciente del entrenamiento con un modelo ajustado en el tramo anterior."""
    ts = np.sort(train["timestamp"].unique())
    cut = ts[int(len(ts) * (1 - validation_fraction))]
    fit, val = train[train["timestamp"] < cut], train[train["timestamp"] >= cut]
    pred = fit_predict(fit, val, cols, target, params, categorical)
    weights = {}
    for node in val["node"].unique():
        m = (val["node"] == node).to_numpy()
        y, p, n = val[target].to_numpy()[m], pred[m], val[lag_col].to_numpy()[m]
        errors = [np.abs(y - (w * p + (1 - w) * n)).sum() for w in BLEND_GRID]
        weights[str(node)] = float(BLEND_GRID[int(np.argmin(errors))])
    return weights


def apply_blend(pred: np.ndarray, nodes, persistence: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    """``w * pred + (1 - w) * persistence`` con el peso de cada fila según su nodo
    (un nodo sin peso conserva la predicción del modelo)."""
    w = pd.Series(np.asarray(nodes)).map(weights).fillna(1.0).to_numpy()
    return w * np.asarray(pred) + (1 - w) * np.asarray(persistence)
