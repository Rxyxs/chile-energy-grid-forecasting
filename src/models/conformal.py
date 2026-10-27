"""Intervalos de predicción con Conformalized Quantile Regression (CQR), para
los 4 targets del SEN.

`train_forecaster.py` solo produce pronósticos puntuales (WAPE/MAE), pero el
README vende el proyecto como una herramienta para "anticipar riesgo de
price-spike" -- eso exige una banda de incertidumbre, no solo un número. Este
módulo la agrega sin tocar el modelo puntual: entrena un par de regresores de
cuantiles LightGBM (`LOWER_Q`/`UPPER_Q`) por target y los conformaliza (Romano,
Patterson & Candès, 2019) con un tramo de calibración temporal separado del
set de ajuste, para que la cobertura empírica del intervalo converja al nivel
nominal aunque los cuantiles crudos del modelo estén mal calibrados.

Reutiliza los folds walk-forward de `validation.get_walk_forward_folds` (los
mismos que usa el modelo puntual) y el `_prepare_fold`/`get_feature_columns`
de `train_forecaster.py` -- ya usados entre módulos pese al guion bajo (ver
`rolling_stability.py`, `generate_torch_report.py`), así que seguir el mismo
patrón acá no introduce una convención nueva.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.data.real_panel import rows_for_target
from src.models.train_forecaster import (
    CATEGORICAL_FEATURES,
    DEFAULT_LGBM_PARAMS,
    METRICS_OUTPUT_PATH,
    TARGET_COLUMNS,
    _prepare_fold,
    get_feature_columns,
    load_features,
)
from src.models.validation import get_walk_forward_folds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "data" / "processed"
CONFORMAL_METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "conformal_metrics.json"

ALPHA = 0.10  # cobertura nominal 90%
LOWER_Q = ALPHA / 2
UPPER_Q = 1 - ALPHA / 2
N_SPLITS = 5
# Fracción MÁS RECIENTE del train walk-forward (no una muestra aleatoria del
# medio) reservada para calibración -- ver `_split_train_calibration`.
CALIBRATION_FRACTION = 0.15

BASE_LGBM_PARAMS = {"random_state": 42, "verbosity": -1}
# Conformal por grupos ("Mondrian"): un margen por hora del día, calibrado solo
# con las horas de calibración de esa misma hora. Se evalúa junto al margen
# global (cobertura media y de la peor hora), pero no se despliega: con datos
# reales no arregla la peor hora de la solar (55% -> 60%) y empeora la de la
# demanda y el precio, porque el desajuste viene sobre todo del cambio de
# estación entre la calibración y el test, no de la hora del día (README §5.5).
# Un grupo con menos filas de calibración que esto usa el margen global.
MIN_GROUP_CALIBRATION = 30
# Adaptive Conformal Inference (Gibbs y Candès, 2021): el nivel del cuantil se
# ajusta después de cada hora observada, alpha_t+1 = alpha_t + gamma (alpha - error_t).
# Corrige el cambio de estación entre la calibración y el uso, que el margen fijo
# no ve (README §5.5, src/models/conformal_shift.py). gamma fijado antes de mirar resultados.
ACI_GAMMA = 0.005


def _quantile_params(base_params: dict, quantile: float) -> dict:
    return {**BASE_LGBM_PARAMS, **base_params, "objective": "quantile", "alpha": quantile}


def _split_train_calibration(train_ts: set, calibration_fraction: float = CALIBRATION_FRACTION) -> tuple[set, set]:
    """Separa el train walk-forward (expansivo) en un tramo de ajuste (`fit_ts`)
    y uno de calibración conformal (`calib_ts`): el último `calibration_fraction`
    de los timestamps de train, en orden cronológico, inmediatamente antes del
    test set.

    Calibrar con el tramo MÁS RECIENTE de train -- no uno aleatorio del medio --
    es lo que hace que la corrección generalice al test, que es aún más
    reciente: calibrar con datos de hace meses mediría cuán mal calibrado
    estaba el modelo entonces, no ahora, y en una serie con estacionalidad esa
    diferencia no es cosmética.
    """
    sorted_ts = np.sort(list(train_ts))
    n_calib = max(1, int(len(sorted_ts) * calibration_fraction))
    return set(sorted_ts[:-n_calib]), set(sorted_ts[-n_calib:])


def cqr_margin(y_calib: np.ndarray, lower_calib: np.ndarray, upper_calib: np.ndarray, alpha: float = ALPHA) -> float:
    """Puntaje de no-conformidad CQR: `score_i = max(lower_pred_i - y_i, y_i - upper_pred_i)`
    -- positivo cuando `y_i` cae fuera del intervalo crudo (y mide cuánto se
    salió), negativo cuando cae adentro (y mide cuánto margen sobraba). El
    margen final es el cuantil `(1-alpha)` de esos puntajes, con la corrección
    de muestra finita `ceil((n+1)(1-alpha))/n`: sin ella, el cuantil muestral
    simple subestima el cuantil poblacional con `n` finito, y el intervalo
    conformalizado cubre sistemáticamente MENOS que el nivel nominal en
    promedio sobre calibraciones repetidas -- no es un detalle cosmético de la
    fórmula, es la diferencia entre una garantía real y una aproximada.
    """
    scores = np.maximum(lower_calib - y_calib, y_calib - upper_calib)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def scores(y, lower, upper) -> np.ndarray:
    """Puntaje de no conformidad CQR: cuánto queda ``y`` fuera del intervalo crudo."""
    return np.maximum(np.asarray(lower) - y, y - np.asarray(upper))


def finite_quantile(s: np.ndarray, alpha: float) -> float:
    """Cuantil conformal ``ceil((n+1)(1-alpha))/n`` (el de ``cqr_margin``)."""
    n = len(s)
    level = min(1.0, max(0.0, np.ceil((n + 1) * (1 - alpha)) / n))
    return float(np.quantile(s, level, method="higher"))


def online_intervals(y, lower, upper, ts, calib_scores, mode: str, alpha: float = ALPHA,
                     gamma: float = ACI_GAMMA) -> tuple[np.ndarray, float]:
    """Margen hora a hora para ``rolling`` y ``aci``. Todas las filas de una misma
    hora (las 6 barras) comparten margen; el valor real de esa hora entra recién
    después de usarla. Devuelve los márgenes por fila y el margen para la hora
    siguiente a la última (el que se despliega)."""
    order = np.argsort(ts, kind="stable")
    y, lower, upper, ts = y[order], lower[order], upper[order], ts[order]
    margins = np.empty(len(y))
    window = list(calib_scores)
    n_window = len(window)
    alpha_t = alpha
    starts = np.r_[0, np.flatnonzero(ts[1:] != ts[:-1]) + 1, len(ts)]
    for a, b in zip(starts[:-1], starts[1:]):
        if mode == "rolling":
            q = finite_quantile(np.asarray(window[-n_window:]), alpha)
        else:  # aci
            q = finite_quantile(calib_scores, alpha_t) if alpha_t > 0 else float(np.max(calib_scores))
        margins[a:b] = q
        s = scores(y[a:b], lower[a:b], upper[a:b])
        if mode == "rolling":
            window.extend(s.tolist())
        else:
            err = float(np.mean(s > q))
            alpha_t = alpha_t + gamma * (alpha - err)
    out = np.empty(len(y))
    out[order] = margins
    if mode == "rolling":
        nxt = finite_quantile(np.asarray(window[-n_window:]), alpha)
    else:
        nxt = finite_quantile(calib_scores, alpha_t) if alpha_t > 0 else float(np.max(calib_scores))
    return out, nxt


def group_margins(y_calib: np.ndarray, lower_calib: np.ndarray, upper_calib: np.ndarray, groups: np.ndarray,
                  alpha: float = ALPHA, min_rows: int = MIN_GROUP_CALIBRATION) -> dict:
    """Margen CQR por grupo (hora del día), con el margen global como respaldo."""
    out = {"global": cqr_margin(y_calib, lower_calib, upper_calib, alpha), "by_group": {}}
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= min_rows:
            out["by_group"][int(g)] = cqr_margin(y_calib[m], lower_calib[m], upper_calib[m], alpha)
    return out


def margin_for(margin, groups) -> np.ndarray:
    """Margen a aplicar en cada fila: un número (margen global) o el dict de
    ``group_margins`` (margen de la hora de cada fila)."""
    groups = np.asarray(groups)
    if not isinstance(margin, dict):
        return np.full(len(groups), float(margin))
    by_group = {int(k): v for k, v in margin["by_group"].items()}
    return np.array([by_group.get(int(g), margin["global"]) for g in groups])


def rectify_crossing(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Los dos regresores de cuantiles se entrenan por separado, así que nada
    les impide "cruzarse" (`lower > upper`) en una fila puntual -- en la
    práctica pasa cerca de 0 en `solar_generation_mwh` (0.46% de las filas del
    último fold, verificado), donde ambos cuantiles predicen valores casi
    idénticos cerca del piso físico de 0 MWh y el orden entre ellos se vuelve
    ruido de estimación. La rectificación estándar (Chernozhukov, Fernández-Val
    & Galichon, 2010) es tomar el mínimo/máximo elemento a elemento -- no
    cambia ningún intervalo que ya estuviera bien ordenado, y convierte los
    pocos que no lo estaban en el intervalo degenerado más angosto posible en
    vez de uno lógicamente inválido."""
    return np.minimum(lower, upper), np.maximum(lower, upper)


def evaluate_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, groups: np.ndarray | None = None) -> dict:
    lower, upper = rectify_crossing(lower, upper)
    covered = (y_true >= lower) & (y_true <= upper)
    out = {"coverage": float(covered.mean()), "mean_width": float(np.mean(upper - lower))}
    if groups is not None:
        by_group = pd.Series(covered).groupby(np.asarray(groups)).mean()
        out["worst_group_coverage"] = float(by_group.min())
    return out


def train_conformal_intervals(
    df: pd.DataFrame,
    target_column: str,
    n_splits: int = N_SPLITS,
    calibration_fraction: float = CALIBRATION_FRACTION,
    alpha: float = ALPHA,
    lgbm_params: dict | None = None,
) -> dict:
    """CQR walk-forward: por fold, separa fit/calibración dentro del train
    expansivo, entrena un par de regresores de cuantiles LightGBM sobre fit,
    conformaliza con calibración, y evalúa sobre el mismo test set que usa el
    modelo puntual (`train_with_walk_forward_validation`).

    Reporta DOS coberturas por fold, no solo la conformalizada: la cruda (los
    cuantiles de LightGBM tal cual, sin corregir) y la conformalizada. Esa
    comparación es la ablación que aísla si conformalizar realmente hace algo
    -- si la cruda ya diera ~90% de cobertura, el paso de calibración sería
    innecesario; si ambas fallan igual de mal, el problema no sería la
    calibración sino el modelo de cuantiles en sí.
    """
    feature_columns = get_feature_columns(df)
    params = lgbm_params or DEFAULT_LGBM_PARAMS
    df = rows_for_target(df, target_column).dropna(subset=feature_columns + [target_column]).reset_index(drop=True)

    fold_metrics = []
    last_lower_model = last_upper_model = None
    last_margin = None

    for fold, (train_ts, test_ts) in enumerate(get_walk_forward_folds(df, n_splits)):
        fit_ts, calib_ts = _split_train_calibration(train_ts, calibration_fraction)

        X_fit, y_fit, X_calib, y_calib = _prepare_fold(df, feature_columns, target_column, fit_ts, calib_ts)
        _, _, X_test, y_test = _prepare_fold(df, feature_columns, target_column, fit_ts, test_ts)

        lower_model = LGBMRegressor(**_quantile_params(params, LOWER_Q))
        upper_model = LGBMRegressor(**_quantile_params(params, UPPER_Q))
        lower_model.fit(X_fit, y_fit, categorical_feature=CATEGORICAL_FEATURES)
        upper_model.fit(X_fit, y_fit, categorical_feature=CATEGORICAL_FEATURES)

        lower_calib, upper_calib = lower_model.predict(X_calib), upper_model.predict(X_calib)
        hour_calib = df.loc[y_calib.index, "timestamp"].dt.hour.to_numpy()
        hour_test = df.loc[y_test.index, "timestamp"].dt.hour.to_numpy()
        margin = cqr_margin(y_calib.to_numpy(), lower_calib, upper_calib, alpha)
        hourly = group_margins(y_calib.to_numpy(), lower_calib, upper_calib, hour_calib, alpha)

        lower_test, upper_test = lower_model.predict(X_test), upper_model.predict(X_test)
        y_test_arr = y_test.to_numpy()
        m_hour = margin_for(hourly, hour_test)

        calib_scores = scores(y_calib.to_numpy(), lower_calib, upper_calib)
        m_aci, next_aci = online_intervals(y_test_arr, lower_test, upper_test,
                                           df.loc[y_test.index, "timestamp"].to_numpy(), calib_scores, "aci", alpha)

        naive = evaluate_coverage(y_test_arr, lower_test, upper_test, hour_test)
        conformal = evaluate_coverage(y_test_arr, lower_test - margin, upper_test + margin, hour_test)
        by_hour = evaluate_coverage(y_test_arr, lower_test - m_hour, upper_test + m_hour, hour_test)
        aci = evaluate_coverage(y_test_arr, lower_test - m_aci, upper_test + m_aci, hour_test)

        fold_metrics.append({
            "fold": fold,
            "fit_rows": int(len(X_fit)), "calib_rows": int(len(X_calib)), "test_rows": int(len(X_test)),
            "margin": margin,
            "naive_coverage": naive["coverage"], "naive_mean_width": naive["mean_width"],
            "naive_worst_hour_coverage": naive["worst_group_coverage"],
            "conformal_coverage": conformal["coverage"], "conformal_mean_width": conformal["mean_width"],
            "conformal_worst_hour_coverage": conformal["worst_group_coverage"],
            "hourly_coverage": by_hour["coverage"], "hourly_mean_width": by_hour["mean_width"],
            "hourly_worst_hour_coverage": by_hour["worst_group_coverage"],
            "aci_coverage": aci["coverage"], "aci_mean_width": aci["mean_width"],
            "aci_worst_hour_coverage": aci["worst_group_coverage"], "aci_next_margin": next_aci,
        })
        # El margen desplegado es el de ACI para la hora siguiente al final de los datos.
        last_lower_model, last_upper_model, last_margin = lower_model, upper_model, next_aci

    return {
        "target_column": target_column,
        "alpha": alpha,
        "nominal_coverage": 1 - alpha,
        "n_splits": n_splits,
        "fold_metrics": fold_metrics,
        "mean_naive_coverage": float(np.mean([m["naive_coverage"] for m in fold_metrics])),
        "mean_conformal_coverage": float(np.mean([m["conformal_coverage"] for m in fold_metrics])),
        "mean_naive_width": float(np.mean([m["naive_mean_width"] for m in fold_metrics])),
        "mean_conformal_width": float(np.mean([m["conformal_mean_width"] for m in fold_metrics])),
        **{f"mean_{k}": float(np.mean([m[k] for m in fold_metrics]))
           for k in ("naive_worst_hour_coverage", "conformal_worst_hour_coverage", "hourly_coverage",
                     "hourly_mean_width", "hourly_worst_hour_coverage", "aci_coverage", "aci_mean_width",
                     "aci_worst_hour_coverage")},
        "worst_fold_conformal_coverage": float(np.min([m["conformal_coverage"] for m in fold_metrics])),
        "worst_fold_aci_coverage": float(np.min([m["aci_coverage"] for m in fold_metrics])),
        "lower_model": last_lower_model,
        "upper_model": last_upper_model,
        "margin": last_margin,
    }


def load_conformal_artifacts() -> dict[str, dict]:
    """Carga, por target, los modelos de cuantiles y el margen conformal
    guardados por `run_full_pipeline` -- el par (`lower_model`, `upper_model`)
    corresponde al último fold, igual que `train_forecaster.load_models`
    guarda el modelo puntual del último fold como el desplegado."""
    if not CONFORMAL_METRICS_PATH.exists():
        raise FileNotFoundError(f"Falta {CONFORMAL_METRICS_PATH} -- corre `python -m src.models.conformal` primero.")
    summary = json.loads(CONFORMAL_METRICS_PATH.read_text(encoding="utf-8"))
    artifacts = {}
    for target, info in summary.items():
        artifacts[target] = {
            "lower_model": joblib.load(MODEL_DIR / f"conformal_{target}_lower.joblib"),
            "upper_model": joblib.load(MODEL_DIR / f"conformal_{target}_upper.joblib"),
            "margin": info["margin"],
        }
    return artifacts


def run_full_pipeline(n_splits: int = N_SPLITS) -> dict:
    """Entrena y conformaliza los intervalos de los 4 targets, reutilizando los
    hiperparámetros ya tuneados por `train_forecaster.run_full_pipeline`
    (`forecaster_metrics.json`) en vez de correr Optuna de nuevo para el
    objetivo de cuantiles -- mantiene fijo todo lo demás y aísla el único
    cambio real: la función de pérdida (quantile/pinball en vez de la default
    de LightGBM), el mismo principio de ablación de un solo cambio a la vez
    que ya usa `notebooks/02_Weather_Augmented_Rolling_CV.ipynb`."""
    if not METRICS_OUTPUT_PATH.exists():
        raise FileNotFoundError(f"Falta {METRICS_OUTPUT_PATH} -- corre `python -m src.models.train_forecaster` primero.")
    point_metrics = json.loads(METRICS_OUTPUT_PATH.read_text(encoding="utf-8"))

    df = load_features()
    all_metrics: dict[str, dict] = {}

    for target in TARGET_COLUMNS:
        print(f"\n=== {target} ===")
        best_params = point_metrics[target]["tuning"]["best_params"]

        result = train_conformal_intervals(df, target, n_splits=n_splits, lgbm_params=best_params)
        for m in result["fold_metrics"]:
            print(
                f"  Fold {m['fold']}: margen={m['margin']:.3f} | "
                f"cobertura cruda={m['naive_coverage']:.1%} (ancho {m['naive_mean_width']:.2f}) | "
                f"cobertura conforme={m['conformal_coverage']:.1%} (ancho {m['conformal_mean_width']:.2f})"
            )
        print(
            f"  Nominal: {result['nominal_coverage']:.0%} | "
            f"Cruda promedio: {result['mean_naive_coverage']:.1%} | "
            f"Conforme promedio: {result['mean_conformal_coverage']:.1%} | "
            f"Por hora: {result['mean_hourly_coverage']:.1%}"
        )
        print(
            f"  Peor hora -- cruda {result['mean_naive_worst_hour_coverage']:.1%} | "
            f"global {result['mean_conformal_worst_hour_coverage']:.1%} (ancho {result['mean_conformal_width']:.1f}) | "
            f"por hora {result['mean_hourly_worst_hour_coverage']:.1%} (ancho {result['mean_hourly_mean_width']:.1f})"
        )
        print(
            f"  ACI -- cobertura {result['mean_aci_coverage']:.1%} | peor fold {result['worst_fold_aci_coverage']:.1%} "
            f"(margen fijo {result['worst_fold_conformal_coverage']:.1%}) | ancho {result['mean_aci_mean_width']:.1f}"
        )

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(result["lower_model"], MODEL_DIR / f"conformal_{target}_lower.joblib")
        joblib.dump(result["upper_model"], MODEL_DIR / f"conformal_{target}_upper.joblib")

        all_metrics[target] = {k: v for k, v in result.items() if k not in ("lower_model", "upper_model")}

    CONFORMAL_METRICS_PATH.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"\nMétricas guardadas en {CONFORMAL_METRICS_PATH}")
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intervalos de predicción CQR (Conformalized Quantile Regression) para los 4 targets del SEN.")
    parser.add_argument("--splits", type=int, default=N_SPLITS, help="Folds de validación walk-forward.")
    args = parser.parse_args()

    run_full_pipeline(n_splits=args.splits)
