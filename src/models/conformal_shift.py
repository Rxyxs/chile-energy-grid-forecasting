"""Intervalos conformales bajo cambio de estación: cuatro formas de calibrar.

Con el margen calibrado en el 15% más reciente del entrenamiento, la cobertura
de la solar va de 73% a 94% según el fold: el margen se calibra en una estación
y se usa en la siguiente. Este experimento compara, con los mismos modelos de
cuantiles, folds y nivel nominal (90%):

- ``recent``: el margen fijo actual (último 15% del entrenamiento).
- ``seasonal``: calibrar con las horas del entrenamiento de la misma época del
  año que el test (±45 días de calendario, de cualquier año previo, incluidas
  las semanas justo antes del test); si son muy pocas, se usa ``recent``.
- ``rolling``: recalibrar hora a hora con los puntajes de las últimas horas ya
  observadas (las del final del entrenamiento y, a medida que se conocen, las
  del test). En operación el valor real de cada hora se conoce una hora
  después, así que no mira el futuro.
- ``aci``: Adaptive Conformal Inference (Gibbs y Candès, 2021): el nivel del
  cuantil se ajusta después de cada hora según si el intervalo cubrió,
  ``alpha_t+1 = alpha_t + gamma (alpha - error_t)``, con ``gamma = 0.005``
  fijado antes de mirar resultados.

Métricas por target: cobertura media, cobertura del peor fold (la que el
cambio de estación arruina), cobertura de la peor hora del día y ancho medio.

    python -m src.models.conformal_shift
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.data.real_panel import rows_for_target
from src.models.conformal import (
    ALPHA, CALIBRATION_FRACTION, LOWER_Q, UPPER_Q, N_SPLITS, _quantile_params, _split_train_calibration,
    finite_quantile, online_intervals, rectify_crossing, scores,
)
from src.models.train_forecaster import (
    CATEGORICAL_FEATURES, METRICS_OUTPUT_PATH, TARGET_COLUMNS, _prepare_fold, get_feature_columns, load_features,
)
from src.models.validation import get_walk_forward_folds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "conformal_shift_metrics.json"
SEASON_WINDOW_DAYS = 45
MIN_SEASONAL_FRACTION = 0.05
VARIANTS = ("recent", "seasonal", "rolling", "aci")


def seasonal_split(train_ts: set, test_ts: set, window_days: int = SEASON_WINDOW_DAYS,
                   min_fraction: float = MIN_SEASONAL_FRACTION) -> tuple[set, set, bool]:
    """Calibración = horas del entrenamiento a menos de ``window_days`` de calendario
    (día del año, circular) de algún día del test. Devuelve (fit, calib, usó_estacional)."""
    train = pd.DatetimeIndex(sorted(train_ts))
    test_doy = np.unique(pd.DatetimeIndex(sorted(test_ts)).dayofyear.to_numpy())
    d = np.abs(train.dayofyear.to_numpy()[:, None] - test_doy[None, :])
    d = np.minimum(d, 366 - d).min(axis=1)
    calib_mask = d <= window_days
    if calib_mask.sum() < min_fraction * len(train):
        fit, calib = _split_train_calibration(train_ts)
        return fit, calib, False
    return set(train[~calib_mask]), set(train[calib_mask]), True


def _fit_quantiles(df, cols, target, fit_ts, params):
    X_fit, y_fit, _, _ = _prepare_fold(df, cols, target, fit_ts, fit_ts)
    lo = LGBMRegressor(**_quantile_params(params, LOWER_Q)).fit(X_fit, y_fit, categorical_feature=CATEGORICAL_FEATURES)
    hi = LGBMRegressor(**_quantile_params(params, UPPER_Q)).fit(X_fit, y_fit, categorical_feature=CATEGORICAL_FEATURES)
    return lo, hi


def _coverage(y, lower, upper, hours) -> dict:
    lower, upper = rectify_crossing(lower, upper)
    covered = (y >= lower) & (y <= upper)
    return {"coverage": float(covered.mean()), "width": float(np.mean(upper - lower)),
            "worst_hour": float(pd.Series(covered).groupby(hours).mean().min())}


def evaluate_target(df: pd.DataFrame, target: str, params: dict, n_splits: int = N_SPLITS) -> dict:
    cols = get_feature_columns(df)
    df = rows_for_target(df, target).dropna(subset=cols + [target]).reset_index(drop=True)
    folds = {v: [] for v in VARIANTS}
    for train_ts, test_ts in get_walk_forward_folds(df, n_splits):
        test = df[df["timestamp"].isin(test_ts)]
        X_test = test[cols].copy()
        for c in CATEGORICAL_FEATURES:
            X_test[c] = X_test[c].astype("category")
        y = test[target].to_numpy()
        ts = test["timestamp"].to_numpy()
        hours = test["timestamp"].dt.hour.to_numpy()

        fit_ts, calib_ts = _split_train_calibration(train_ts, CALIBRATION_FRACTION)
        lo, hi = _fit_quantiles(df, cols, target, fit_ts, params)
        _, _, X_cal, y_cal = _prepare_fold(df, cols, target, fit_ts, calib_ts)
        cal_scores = scores(y_cal.to_numpy(), lo.predict(X_cal), hi.predict(X_cal))
        lower, upper = lo.predict(X_test), hi.predict(X_test)
        m = finite_quantile(cal_scores, ALPHA)
        folds["recent"].append(_coverage(y, lower - m, upper + m, hours))
        for mode in ("rolling", "aci"):
            mm, _ = online_intervals(y, lower, upper, ts, cal_scores, mode)
            folds[mode].append(_coverage(y, lower - mm, upper + mm, hours))

        fit_s, calib_s, used = seasonal_split(train_ts, test_ts)
        lo_s, hi_s = _fit_quantiles(df, cols, target, fit_s, params)
        _, _, X_cs, y_cs = _prepare_fold(df, cols, target, fit_s, calib_s)
        ms = finite_quantile(scores(y_cs.to_numpy(), lo_s.predict(X_cs), hi_s.predict(X_cs)), ALPHA)
        res = _coverage(y, lo_s.predict(X_test) - ms, hi_s.predict(X_test) + ms, hours)
        res["seasonal_used"] = used
        folds["seasonal"].append(res)

    summary = {}
    for v, rows in folds.items():
        cov = [r["coverage"] for r in rows]
        summary[v] = {"mean_coverage": float(np.mean(cov)), "worst_fold_coverage": float(np.min(cov)),
                      "fold_coverage": cov, "mean_worst_hour": float(np.mean([r["worst_hour"] for r in rows])),
                      "mean_width": float(np.mean([r["width"] for r in rows]))}
    summary["seasonal"]["folds_with_seasonal_calibration"] = int(sum(r["seasonal_used"] for r in folds["seasonal"]))
    return summary


def run() -> dict:
    tuned = json.loads(METRICS_OUTPUT_PATH.read_text(encoding="utf-8"))
    df = load_features()
    out = {}
    for target in TARGET_COLUMNS:
        print(f"=== {target} ===")
        out[target] = evaluate_target(df, target, tuned[target]["tuning"]["best_params"])
        for v in VARIANTS:
            s = out[target][v]
            print(f"  {v:9s} cobertura {s['mean_coverage']:.1%} | peor fold {s['worst_fold_coverage']:.1%} | "
                  f"peor hora {s['mean_worst_hour']:.1%} | ancho {s['mean_width']:.1f}")
    OUTPUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    run()
