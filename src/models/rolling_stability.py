"""Evaluación de estabilidad temporal de LightGBM vs. XGBoost vía Rolling-
Origin CV (ventana deslizante fija, `validation.get_rolling_origin_folds`) --
a diferencia de la validación walk-forward expansiva de `train_forecaster.py`
(usada para entrenar el modelo final con el máximo de datos disponible), acá
el objetivo es medir si el error de cada modelo se mantiene estable bloque a
bloque cuando siempre entrena con la misma cantidad de historial reciente, no
maximizar su precisión puntual.

XGBoost no se usaba en ningún punto del proyecto pese a estar en
`requirements.txt` -- este módulo lo agrega específicamente como segundo
modelo de comparación para esta evaluación de estabilidad, entrenado con las
mismas columnas categóricas nativas (`enable_categorical=True`, igual que
`categorical_feature` en LightGBM) para que la comparación sea justa.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from src.models.train_forecaster import CATEGORICAL_FEATURES, TARGET_COLUMNS, get_feature_columns, load_features
from src.models.validation import get_rolling_origin_folds, wape

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STABILITY_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "temporal_stability_metrics.json"

DEFAULT_N_SPLITS = 8
DEFAULT_TRAIN_WINDOW_DAYS = 180
DEFAULT_TEST_WINDOW_DAYS = 14

LGBM_PARAMS = {
    "random_state": 42, "verbosity": -1,
    "n_estimators": 400, "learning_rate": 0.05, "max_depth": 7, "num_leaves": 63,
}
XGB_PARAMS = {
    "random_state": 42, "n_estimators": 400, "learning_rate": 0.05, "max_depth": 7,
    "tree_method": "hist", "enable_categorical": True,
}


def _prepare_fold(
    df: pd.DataFrame, feature_columns: list[str], target_column: str, train_ts: set, test_ts: set
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_mask = df["timestamp"].isin(train_ts)
    test_mask = df["timestamp"].isin(test_ts)

    X_train, y_train = df.loc[train_mask, feature_columns].copy(), df.loc[train_mask, target_column]
    X_test, y_test = df.loc[test_mask, feature_columns].copy(), df.loc[test_mask, target_column]

    for col in CATEGORICAL_FEATURES:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    return X_train, y_train, X_test, y_test


def evaluate_temporal_stability(
    df: pd.DataFrame,
    target_column: str,
    n_splits: int = DEFAULT_N_SPLITS,
    train_window_days: int = DEFAULT_TRAIN_WINDOW_DAYS,
    test_window_days: int = DEFAULT_TEST_WINDOW_DAYS,
    lgbm_params: dict | None = None,
    xgb_params: dict | None = None,
) -> dict:
    """Entrena LightGBM y XGBoost sobre los mismos folds Rolling-Origin (ver
    `validation.get_rolling_origin_folds`) y reporta, por modelo: el WAPE de
    cada fold, y su media/desvío/coeficiente de variación entre folds --
    la métrica de "estabilidad temporal" en sí: un CV bajo significa que el
    modelo generaliza parecido sin importar en qué bloque de 2024-2025 se
    lo evalúe; un CV alto señala que el modelo (o el fenómeno) es sensible
    al período de entrenamiento/evaluación.
    """
    feature_columns = get_feature_columns(df)
    clean_df = df.dropna(subset=feature_columns + [target_column]).reset_index(drop=True)

    train_window_hours = train_window_days * 24
    test_window_hours = test_window_days * 24
    folds = get_rolling_origin_folds(clean_df, n_splits, train_window_hours, test_window_hours)

    lgbm_p = {**LGBM_PARAMS, **(lgbm_params or {})}
    xgb_p = {**XGB_PARAMS, **(xgb_params or {})}

    fold_results: dict[str, list[dict]] = {"lightgbm": [], "xgboost": []}
    for fold, (train_ts, test_ts) in enumerate(folds):
        X_train, y_train, X_test, y_test = _prepare_fold(clean_df, feature_columns, target_column, train_ts, test_ts)
        y_test_np = y_test.to_numpy()

        lgbm_model = LGBMRegressor(**lgbm_p)
        lgbm_model.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)
        lgbm_pred = lgbm_model.predict(X_test)

        xgb_model = XGBRegressor(**xgb_p)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)

        fold_window = {
            "fold": fold,
            "test_start": str(min(test_ts)),
            "test_end": str(max(test_ts)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        }
        fold_results["lightgbm"].append({
            **fold_window, "wape": wape(y_test_np, lgbm_pred), "mae": float(mean_absolute_error(y_test_np, lgbm_pred)),
        })
        fold_results["xgboost"].append({
            **fold_window, "wape": wape(y_test_np, xgb_pred), "mae": float(mean_absolute_error(y_test_np, xgb_pred)),
        })

    summary: dict[str, dict] = {}
    for model_name, metrics in fold_results.items():
        wapes = np.array([m["wape"] for m in metrics])
        mean_wape = float(wapes.mean())
        std_wape = float(wapes.std(ddof=0))
        summary[model_name] = {
            "fold_metrics": metrics,
            "mean_wape": mean_wape,
            "std_wape": std_wape,
            "cv_wape": std_wape / mean_wape if mean_wape != 0 else float("nan"),
            "min_wape": float(wapes.min()),
            "max_wape": float(wapes.max()),
        }

    return {
        "target_column": target_column,
        "n_splits": n_splits,
        "train_window_days": train_window_days,
        "test_window_days": test_window_days,
        **summary,
    }


def run_stability_evaluation(
    n_splits: int = DEFAULT_N_SPLITS,
    train_window_days: int = DEFAULT_TRAIN_WINDOW_DAYS,
    test_window_days: int = DEFAULT_TEST_WINDOW_DAYS,
) -> dict:
    df = load_features()
    all_results = {}
    for target in TARGET_COLUMNS:
        print(f"\n=== Estabilidad temporal: {target} ===")
        result = evaluate_temporal_stability(df, target, n_splits, train_window_days, test_window_days)
        for model_name in ("lightgbm", "xgboost"):
            s = result[model_name]
            print(f"  {model_name:10s} -> WAPE medio: {s['mean_wape']:.2f}% | desvío: {s['std_wape']:.2f} | CV: {s['cv_wape']:.3f}")
        all_results[target] = result

    STABILITY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    STABILITY_OUTPUT_PATH.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nMétricas de estabilidad guardadas en {STABILITY_OUTPUT_PATH}")
    return all_results


if __name__ == "__main__":
    run_stability_evaluation()
