"""Orquesta el 3er enfoque de modelado end-to-end para el target por defecto
(`marginal_cost_usd_mwh`, el mismo que usa `rolling_stability.py` para la
comparación LightGBM-vs-XGBoost): entrena el MLP PyTorch con las 3
activaciones, genera los gráficos explicativos (predicho vs. real, residuos,
loss/época, comparación de activaciones) y persiste una tabla comparativa
naive / seasonal-naive / LightGBM / XGBoost / MLP-PyTorch en DuckDB.

Ejecutar con: `python -m src.models.generate_torch_report`
"""
from __future__ import annotations

import time

import numpy as np

from src.models.baselines import naive_metrics, seasonal_naive_metrics
from src.models.metrics_db import get_connection, record_comparison_row
from src.models.rolling_stability import LGBM_PARAMS, XGB_PARAMS, _prepare_fold as _tree_prepare_fold
from src.models.torch_forecaster import DEFAULT_TARGET, N_SPLITS, compare_activations, predict_mlp
from src.models.torch_plots import plot_activation_comparison, plot_loss_curve, plot_predicted_vs_actual, plot_residuals
from src.models.train_forecaster import get_feature_columns, load_features
from src.models.validation import get_walk_forward_folds, wape
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


def _tree_wape_mae(df, target_column: str, n_splits: int, model_cls, params: dict) -> dict:
    """Reentrena LightGBM/XGBoost sobre los mismos folds walk-forward que el
    MLP -- necesario para la fila de la tabla comparativa en DuckDB, ya que
    `train_forecaster`/`rolling_stability` corren tuning Optuna (lento) por
    defecto y este reporte solo necesita el WAPE/MAE comparable, no el
    modelo tuneado final."""
    feature_columns = get_feature_columns(df)
    clean_df = df.dropna(subset=feature_columns + [target_column]).reset_index(drop=True)

    fold_metrics = []
    for train_ts, test_ts in get_walk_forward_folds(clean_df, n_splits):
        X_train, y_train, X_test, y_test = _tree_prepare_fold(clean_df, feature_columns, target_column, train_ts, test_ts)
        model = model_cls(**params)
        if model_cls is LGBMRegressor:
            model.fit(X_train, y_train, categorical_feature=["node"])
        else:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fold_metrics.append({"wape": wape(y_test.to_numpy(), y_pred), "mae": float(mean_absolute_error(y_test, y_pred))})

    return {
        "mean_wape": float(np.mean([m["wape"] for m in fold_metrics])),
        "mean_mae": float(np.mean([m["mae"] for m in fold_metrics])),
    }


def run(target_column: str = DEFAULT_TARGET, n_splits: int = N_SPLITS, epochs: int = 40) -> None:
    df = load_features()

    print(f"=== Reporte MLP PyTorch -- {target_column} ===")
    print("Entrenando ReLU / GELU / Swish (mismos folds walk-forward)...")
    activation_results = compare_activations(df, target_column, n_splits=n_splits, epochs=epochs)

    best_activation = min(activation_results, key=lambda a: activation_results[a]["mean_wape"])
    best = activation_results[best_activation]
    print(f"Mejor activación: {best_activation} (WAPE promedio {best['mean_wape']:.2f}%)")

    feature_columns = get_feature_columns(df)
    clean_df = df.dropna(subset=feature_columns + [target_column]).reset_index(drop=True)
    last_train_ts, last_test_ts = list(get_walk_forward_folds(clean_df, n_splits))[-1]
    test_mask = clean_df["timestamp"].isin(last_test_ts)
    encoded_test = clean_df.loc[test_mask, feature_columns]
    import pandas as pd
    encoded_test = pd.get_dummies(encoded_test, columns=["node"]).astype("float32")
    encoded_test = encoded_test.reindex(columns=best["feature_columns"], fill_value=0.0)
    y_true = clean_df.loc[test_mask, target_column].to_numpy()
    y_pred = predict_mlp(best["final_model"], encoded_test, best["final_norm_params"])

    print("Generando gráficos...")
    plot_predicted_vs_actual(y_true, y_pred, target_column)
    plot_residuals(y_true, y_pred, target_column)
    plot_loss_curve(best["final_history"], target_column, best_activation)
    plot_activation_comparison(activation_results, target_column)

    print("Persistiendo comparación en DuckDB...")
    con = get_connection()
    try:
        naive = naive_metrics(df, target_column, n_splits=n_splits)
        seasonal = seasonal_naive_metrics(df, target_column, n_splits=n_splits)
        record_comparison_row(con, target_column, "naive", naive["mean_wape"], naive["mean_mae"], n_splits)
        record_comparison_row(con, target_column, "seasonal_naive", seasonal["mean_wape"], seasonal["mean_mae"], n_splits)

        lgbm_metrics = _tree_wape_mae(df, target_column, n_splits, LGBMRegressor, LGBM_PARAMS)
        record_comparison_row(con, target_column, "lightgbm", lgbm_metrics["mean_wape"], lgbm_metrics["mean_mae"], n_splits)

        xgb_metrics = _tree_wape_mae(df, target_column, n_splits, XGBRegressor, XGB_PARAMS)
        record_comparison_row(con, target_column, "xgboost", xgb_metrics["mean_wape"], xgb_metrics["mean_mae"], n_splits)

        for activation, result in activation_results.items():
            start = time.perf_counter()
            predict_mlp(result["final_model"], encoded_test, result["final_norm_params"])
            latency_ms_per_1k = (time.perf_counter() - start) * 1000 / max(len(encoded_test), 1) * 1000
            record_comparison_row(
                con, target_column, "mlp_pytorch", result["mean_wape"], result["mean_mae"], n_splits,
                model_variant=f"mlp_pytorch_{activation}", latency_ms_per_1k_rows=latency_ms_per_1k,
            )
    finally:
        con.close()

    print("Listo.")


if __name__ == "__main__":
    run()
