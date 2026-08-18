"""Entrenamiento del pronosticador (LightGBM) con validación Walk-Forward
(TimeSeriesSplit) y métricas WAPE/MAE por fold.

WAPE (Weighted Absolute Percentage Error) en vez de MAPE: `solar_generation_mwh`
es exactamente 0 de noche, así que un MAPE por punto sería indefinido/infinito en
cada hora nocturna. WAPE agrega antes de dividir (`sum(|err|) / sum(|actual|)`),
así que es robusto a ceros y sigue siendo interpretable como "% de error".
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "sen_features.parquet"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "forecaster_marginal_cost.joblib"
METRICS_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "forecaster_metrics.json"

DEFAULT_TARGET = "marginal_cost_usd_mwh"
# "node" NO va acá: es un identificador de fila, pero también es la feature
# categórica que le permite al modelo distinguir el comportamiento de cada barra.
ID_COLUMNS = ["timestamp"]
TARGET_COLUMNS = ["solar_generation_mwh", "wind_generation_mwh", "demand_mwh", "marginal_cost_usd_mwh"]
CATEGORICAL_FEATURES = ["node"]
N_SPLITS = 5


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (%), robusto a valores reales iguales a 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom * 100)


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.sort_values(["timestamp", "node"]).reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Todas las columnas salvo identificadores y los 4 targets crudos -- así al
    pronosticar `marginal_cost_usd_mwh` no se usa, por ejemplo, `demand_mwh` de
    la misma hora (que en producción tampoco se conoce con certeza de antemano),
    solo sus lags/rolling ya calculados."""
    exclude = set(ID_COLUMNS) | set(TARGET_COLUMNS)
    return [c for c in df.columns if c not in exclude]


def train_with_walk_forward_validation(
    df: pd.DataFrame, target_column: str = DEFAULT_TARGET, n_splits: int = N_SPLITS
) -> dict:
    """Valida con `TimeSeriesSplit` sobre los **timestamps únicos**, no sobre las
    filas crudas del panel: con 5 nodos por hora, un split por fila podría dejar
    la hora 100 de Quillota en train y la hora 99 de Crucero en test -- eso sigue
    siendo lookahead bias aunque el índice de fila sea "posterior". Cada fold
    mueve todos los nodos de un bloque temporal a la vez.
    """
    feature_columns = get_feature_columns(df)
    df = df.dropna(subset=feature_columns + [target_column]).reset_index(drop=True)

    unique_timestamps = np.sort(df["timestamp"].unique())
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_metrics = []
    last_model = None
    for fold, (train_ts_idx, test_ts_idx) in enumerate(tscv.split(unique_timestamps)):
        train_ts = set(unique_timestamps[train_ts_idx])
        test_ts = set(unique_timestamps[test_ts_idx])

        train_mask = df["timestamp"].isin(train_ts)
        test_mask = df["timestamp"].isin(test_ts)

        X_train, y_train = df.loc[train_mask, feature_columns].copy(), df.loc[train_mask, target_column]
        X_test, y_test = df.loc[test_mask, feature_columns].copy(), df.loc[test_mask, target_column]

        for col in CATEGORICAL_FEATURES:
            X_train[col] = X_train[col].astype("category")
            X_test[col] = X_test[col].astype("category")

        model = LGBMRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=7, num_leaves=63,
            random_state=42, verbosity=-1,
        )
        model.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)
        y_pred = model.predict(X_test)

        fold_metrics.append({
            "fold": fold,
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "wape": wape(y_test.to_numpy(), y_pred),
            "mae": float(mean_absolute_error(y_test, y_pred)),
        })
        last_model = model

    return {
        "target_column": target_column,
        "feature_columns": feature_columns,
        "categorical_features": CATEGORICAL_FEATURES,
        "n_splits": n_splits,
        "fold_metrics": fold_metrics,
        "mean_wape": float(np.mean([m["wape"] for m in fold_metrics])),
        "mean_mae": float(np.mean([m["mae"] for m in fold_metrics])),
        # Se despliega el modelo del último fold (entrenado con el tramo más
        # reciente de datos, y validado contra el bloque siguiente) en vez de
        # reentrenar con todo el histórico -- así el modelo que se guarda es
        # exactamente uno de los que ya se evaluó, no uno sin evaluación propia.
        "final_model": last_model,
    }


if __name__ == "__main__":
    df = load_features()
    result = train_with_walk_forward_validation(df)

    print(f"Target: {result['target_column']}")
    print(f"Features: {len(result['feature_columns'])}")
    for m in result["fold_metrics"]:
        print(
            f"  Fold {m['fold']}: train={m['train_rows']:,} test={m['test_rows']:,} "
            f"WAPE={m['wape']:.2f}% MAE={m['mae']:.2f}"
        )
    print(f"\nWAPE promedio: {result['mean_wape']:.2f}%")
    print(f"MAE promedio: {result['mean_mae']:.2f}")

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["final_model"], MODEL_OUTPUT_PATH)

    metrics_out = {k: v for k, v in result.items() if k != "final_model"}
    METRICS_OUTPUT_PATH.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

    print(f"\nModelo guardado en {MODEL_OUTPUT_PATH}")
    print(f"Métricas guardadas en {METRICS_OUTPUT_PATH}")
