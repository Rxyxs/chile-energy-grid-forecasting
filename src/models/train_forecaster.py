"""Entrenamiento de pronosticadores LightGBM -- uno por target (solar, eólica,
demanda y precio) -- con tuning de hiperparámetros vía Optuna y validación
Walk-Forward (TimeSeriesSplit), reportando WAPE/MAE por fold.

Los 4 targets comparten exactamente el mismo set de features (columnas
identificadoras y los 4 targets crudos se excluyen siempre, ver
`get_feature_columns`), así que un mismo pipeline de tuning + validación
sirve para los cuatro sin duplicar lógica.

WAPE (Weighted Absolute Percentage Error) en vez de MAPE: `solar_generation_mwh`
es exactamente 0 de noche, así que un MAPE por punto sería indefinido/infinito en
cada hora nocturna. WAPE agrega antes de dividir (`sum(|err|) / sum(|actual|)`),
así que es robusto a ceros y sigue siendo interpretable como "% de error".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

from src.features.build_features import WEATHER_COLUMNS
from src.models.baselines import naive_metrics, seasonal_naive_metrics
from src.models.validation import get_walk_forward_folds, wape

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "sen_features.parquet"
MODEL_DIR = PROJECT_ROOT / "data" / "processed"
METRICS_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "forecaster_metrics.json"

DEFAULT_TARGET = "marginal_cost_usd_mwh"
# "node" NO va en ID_COLUMNS: es un identificador de fila, pero también es la
# feature categórica que le permite al modelo distinguir el comportamiento de
# cada barra.
ID_COLUMNS = ["timestamp"]
TARGET_COLUMNS = ["solar_generation_mwh", "wind_generation_mwh", "demand_mwh", "marginal_cost_usd_mwh"]
CATEGORICAL_FEATURES = ["node"]
N_SPLITS = 5
N_TUNING_SPLITS = 3
N_TUNING_TRIALS = 20

BASE_LGBM_PARAMS = {"random_state": 42, "verbosity": -1}
DEFAULT_LGBM_PARAMS = {
    "n_estimators": 400, "learning_rate": 0.05, "max_depth": 7, "num_leaves": 63,
}


def load_features(path: Path = FEATURES_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.sort_values(["timestamp", "node"]).reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Todas las columnas salvo identificadores, los 4 targets crudos, y las
    variables meteorológicas crudas de la hora actual -- así al pronosticar,
    por ejemplo, `marginal_cost_usd_mwh` no se usa `demand_mwh` de la misma
    hora (que en producción tampoco se conoce con certeza de antemano), solo
    sus lags/rolling ya calculados. El mismo motivo excluye `ghi_w_m2`,
    `wind_speed_ms` y `temperature_c` crudos (ver
    `build_features.add_weather_lag_features`): solo sus lags entran como
    feature. Idéntico para los 4 targets y las 3 variables climáticas."""
    exclude = set(ID_COLUMNS) | set(TARGET_COLUMNS) | set(WEATHER_COLUMNS)
    return [c for c in df.columns if c not in exclude]


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


def train_with_walk_forward_validation(
    df: pd.DataFrame,
    target_column: str = DEFAULT_TARGET,
    n_splits: int = N_SPLITS,
    lgbm_params: dict | None = None,
) -> dict:
    """Entrena y valida un `LGBMRegressor` con walk-forward CV sobre timestamps
    únicos (ver `validation.get_walk_forward_folds`). El modelo final desplegado
    es el del último fold (entrenado con el tramo más reciente de datos y
    validado contra el bloque siguiente) en vez de reentrenar con todo el
    histórico -- así el modelo que se guarda es exactamente uno de los que ya
    se evaluó, no uno sin evaluación propia.
    """
    feature_columns = get_feature_columns(df)
    df = df.dropna(subset=feature_columns + [target_column]).reset_index(drop=True)
    params = {**BASE_LGBM_PARAMS, **(lgbm_params or DEFAULT_LGBM_PARAMS)}

    fold_metrics = []
    last_model = None
    for fold, (train_ts, test_ts) in enumerate(get_walk_forward_folds(df, n_splits)):
        X_train, y_train, X_test, y_test = _prepare_fold(df, feature_columns, target_column, train_ts, test_ts)

        model = LGBMRegressor(**params)
        model.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)
        y_pred = model.predict(X_test)

        fold_metrics.append({
            "fold": fold,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "wape": wape(y_test.to_numpy(), y_pred),
            "mae": float(mean_absolute_error(y_test, y_pred)),
        })
        last_model = model

    return {
        "target_column": target_column,
        "feature_columns": feature_columns,
        "categorical_features": CATEGORICAL_FEATURES,
        "n_splits": n_splits,
        "lgbm_params": {k: v for k, v in params.items() if k != "random_state"},
        "fold_metrics": fold_metrics,
        "mean_wape": float(np.mean([m["wape"] for m in fold_metrics])),
        "mean_mae": float(np.mean([m["mae"] for m in fold_metrics])),
        "final_model": last_model,
    }


def tune_hyperparameters(
    df: pd.DataFrame,
    target_column: str,
    n_trials: int = N_TUNING_TRIALS,
    n_splits: int = N_TUNING_SPLITS,
    seed: int = 42,
) -> dict:
    """Optuna minimiza el WAPE promedio walk-forward (usando menos folds que la
    validación final, `n_splits` en vez de `N_SPLITS`, para que el tuning corra
    en tiempo razonable) sobre el espacio de hiperparámetros del LightGBM.
    """
    feature_columns = get_feature_columns(df)
    tuning_df = df.dropna(subset=feature_columns + [target_column]).reset_index(drop=True)
    folds = get_walk_forward_folds(tuning_df, n_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {
            **BASE_LGBM_PARAMS,
            "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

        fold_wapes = []
        for train_ts, test_ts in folds:
            X_train, y_train, X_test, y_test = _prepare_fold(tuning_df, feature_columns, target_column, train_ts, test_ts)
            model = LGBMRegressor(**params)
            model.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)
            fold_wapes.append(wape(y_test.to_numpy(), model.predict(X_test)))
        return float(np.mean(fold_wapes))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return {"best_params": study.best_params, "best_tuning_wape": study.best_value, "n_trials": n_trials}


def run_full_pipeline(n_trials: int = N_TUNING_TRIALS, n_splits: int = N_SPLITS) -> dict:
    """Tunea, entrena y valida un modelo por target; agrega la comparación
    contra los baselines naive/seasonal-naive (mismos folds, misma métrica)."""
    df = load_features()
    all_metrics: dict[str, dict] = {}

    for target in TARGET_COLUMNS:
        print(f"\n=== {target} ===")

        print(f"Tuning ({n_trials} trials, {N_TUNING_SPLITS} folds)...")
        tuning = tune_hyperparameters(df, target, n_trials=n_trials)
        print(f"  Mejor WAPE de tuning: {tuning['best_tuning_wape']:.2f}%")

        result = train_with_walk_forward_validation(df, target, n_splits=n_splits, lgbm_params=tuning["best_params"])
        for m in result["fold_metrics"]:
            print(f"  Fold {m['fold']}: train={m['train_rows']:,} test={m['test_rows']:,} WAPE={m['wape']:.2f}% MAE={m['mae']:.2f}")
        print(f"  LightGBM  -> WAPE promedio: {result['mean_wape']:.2f}% | MAE promedio: {result['mean_mae']:.2f}")

        naive = naive_metrics(df, target, n_splits=n_splits)
        seasonal = seasonal_naive_metrics(df, target, n_splits=n_splits)
        print(f"  Naive     -> WAPE promedio: {naive['mean_wape']:.2f}% | MAE promedio: {naive['mean_mae']:.2f}")
        print(f"  Seasonal  -> WAPE promedio: {seasonal['mean_wape']:.2f}% | MAE promedio: {seasonal['mean_mae']:.2f}")

        model_path = MODEL_DIR / f"forecaster_{target}.joblib"
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(result["final_model"], model_path)
        print(f"  Modelo guardado en {model_path}")

        all_metrics[target] = {
            "tuning": tuning,
            "lightgbm": {k: v for k, v in result.items() if k != "final_model"},
            "naive_baseline": naive,
            "seasonal_naive_baseline": seasonal,
        }

    METRICS_OUTPUT_PATH.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"\nMétricas guardadas en {METRICS_OUTPUT_PATH}")
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena pronosticadores LightGBM (tuning Optuna + walk-forward CV) para los 4 targets del SEN.")
    parser.add_argument("--trials", type=int, default=N_TUNING_TRIALS, help="Trials de Optuna por target.")
    parser.add_argument("--splits", type=int, default=N_SPLITS, help="Folds de validación walk-forward final.")
    args = parser.parse_args()

    run_full_pipeline(n_trials=args.trials, n_splits=args.splits)
