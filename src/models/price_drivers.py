"""¿Qué le falta al modelo de precio en las barras donde pierde contra la persistencia?

El modelo de precio (``train_forecaster``) es uno solo para las 6 barras y ve
solo la solar, la eólica y la demanda del sistema. En Puerto Montt pierde
contra el precio de hace una hora: su precio casi no se relaciona con la
demanda residual del sistema. Este experimento prueba, con los mismos folds
walk-forward e hiperparámetros fijos (para no confundirlo con un reajuste):

- **hidro y térmica del sistema** rezagadas (1h, 24h, 168h): la generación
  horaria por tecnología ya está extraída de los informes diarios y no se usaba;
  el sur depende de la hidro;
- **precios rezagados de las 6 barras** (1h y 24h) en cada fila: el estado de la
  red, no solo el de la barra;
- un **modelo por barra** en vez de uno compartido;
- pronosticar el **cambio** respecto de la hora anterior en vez del nivel;
- **mezclar modelo y persistencia** por barra, con el peso elegido en el último
  15% del entrenamiento de cada fold (nunca en el test).

    python -m src.models.price_drivers
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.data import cen_daily_reports as reports
from src.data.real_panel import BAR_NODES, PRICE_TARGET, TECH_COLUMNS, rows_for_target
from src.models.baselines import naive_metrics
from src.models.blend import apply_blend, blend_weights
from src.models.train_forecaster import (
    BASE_LGBM_PARAMS, CATEGORICAL_FEATURES, DEFAULT_LGBM_PARAMS, N_SPLITS, get_feature_columns, load_features,
)
from src.models.validation import get_walk_forward_folds, wape

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "price_drivers_metrics.json"
DRIVER_LAGS = [1, 24, 168]
CROSS_LAGS = [1, 24]
PARAMS = {**BASE_LGBM_PARAMS, **DEFAULT_LGBM_PARAMS}


def hydro_thermal_lags() -> pd.DataFrame:
    """Hidro y térmica del sistema por hora, rezagadas, una fila por timestamp."""
    g = pd.read_csv(reports.GEN_HOURLY_PATH)
    for col in TECH_COLUMNS:
        if col not in g:
            g[col] = 0.0
    g[TECH_COLUMNS] = g[TECH_COLUMNS].fillna(0.0)
    ts = pd.to_datetime(g["date"]) + pd.to_timedelta(g["hour"] - 1, unit="h")
    s = pd.DataFrame({"hydro_mwh": g["hidraulica"].to_numpy(), "thermal_mwh": g["termica"].to_numpy()}, index=ts)
    s = s[~s.index.duplicated()].asfreq("h")
    return pd.DataFrame({f"{c}_lag_{h}h": s[c].shift(h) for c in s.columns for h in DRIVER_LAGS}).rename_axis("timestamp")


def cross_bus_lags(price_rows: pd.DataFrame) -> pd.DataFrame:
    """Precio de las 6 barras hace 1h y 24h, una fila por timestamp."""
    wide = price_rows.pivot_table(index="timestamp", columns="node", values=PRICE_TARGET).asfreq("h")
    short = {n: n.replace("_220kV", "").lower() for n in wide.columns}
    return pd.DataFrame({f"cmg_{short[n]}_lag_{h}h": wide[n].shift(h) for n in wide.columns for h in CROSS_LAGS})


LAG1 = f"{PRICE_TARGET}_lag_1h"
def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], delta: bool = False) -> np.ndarray:
    X_tr, X_te = train[cols].copy(), test[cols].copy()
    cats = [c for c in CATEGORICAL_FEATURES if c in cols]
    for c in cats:
        X_tr[c] = X_tr[c].astype("category")
        X_te[c] = X_te[c].astype("category")
    model = LGBMRegressor(**PARAMS)
    y = train[PRICE_TARGET] - train[LAG1] if delta else train[PRICE_TARGET]
    model.fit(X_tr, y, categorical_feature=cats or "auto")
    pred = model.predict(X_te)
    return pred + test[LAG1].to_numpy() if delta else pred


def evaluate(df: pd.DataFrame, variants: dict[str, list[str]], per_bus: set[str] = frozenset(),
             delta: set[str] = frozenset(), blend: set[str] = frozenset(), n_splits: int = N_SPLITS) -> dict:
    folds = get_walk_forward_folds(df, n_splits)
    out, weights_log = {}, {}
    for name, cols in variants.items():
        preds = []
        for train_ts, test_ts in folds:
            train, test = df[df["timestamp"].isin(train_ts)], df[df["timestamp"].isin(test_ts)]
            if name in per_bus:
                p = pd.Series(np.nan, index=test.index)
                for node in test["node"].unique():
                    tr, te = train[train["node"] == node], test[test["node"] == node]
                    p.loc[te.index] = _fit_predict(tr, te, [c for c in cols if c != "node"])
                pred = p.to_numpy()
            elif name in blend:
                w = blend_weights(train, cols, PRICE_TARGET, PARAMS, CATEGORICAL_FEATURES, LAG1)
                weights_log.setdefault(name, []).append(w)
                pred = apply_blend(_fit_predict(train, test, cols), test["node"], test[LAG1].to_numpy(), w)
            else:
                pred = _fit_predict(train, test, cols, delta=name in delta)
            preds.append(pd.DataFrame({"node": test["node"].to_numpy(), "y": test[PRICE_TARGET].to_numpy(), "p": pred}))
        fold_wape = [wape(f["y"], f["p"]) for f in preds]
        by_node = {n: float(np.mean([wape(f.loc[f["node"] == n, "y"], f.loc[f["node"] == n, "p"]) for f in preds]))
                   for n in BAR_NODES.values()}
        out[name] = {"mean_wape": float(np.mean(fold_wape)), "wape_by_node": by_node, "n_features": len(cols)}
        if name in weights_log:
            out[name]["model_weight_by_fold"] = weights_log[name]
        print(f"  {name:32s} WAPE {out[name]['mean_wape']:6.2f}%  Puerto Montt {by_node['Puerto_Montt_220kV']:6.2f}%")
    return out


def run() -> dict:
    features = load_features()
    base_cols = get_feature_columns(features)
    price = rows_for_target(features, PRICE_TARGET)
    extra = hydro_thermal_lags().join(cross_bus_lags(price), how="outer")
    df = price.merge(extra, left_on="timestamp", right_index=True, how="left")
    ht_cols = [c for c in extra.columns if c.startswith(("hydro", "thermal"))]
    cross_cols = [c for c in extra.columns if c.startswith("cmg_")]
    df = df.dropna(subset=base_cols + ht_cols + cross_cols + [PRICE_TARGET]).reset_index(drop=True)

    variants = {
        "pooled (current features)": base_cols,
        "pooled + hydro/thermal": base_cols + ht_cols,
        "pooled + other buses' prices": base_cols + cross_cols,
        "pooled + both": base_cols + ht_cols + cross_cols,
        "per bus + both": base_cols + ht_cols + cross_cols,
        "pooled, change from 1h ago": base_cols,
        "pooled, blended with persistence": base_cols,
    }
    print(f"{len(df):,} filas de precio, {df['timestamp'].nunique():,} horas")
    results = evaluate(df, variants, per_bus={"per bus + both"}, delta={"pooled, change from 1h ago"},
                       blend={"pooled, blended with persistence"})
    naive = naive_metrics(df, PRICE_TARGET, n_splits=N_SPLITS)
    results["naive (1h persistence)"] = {"mean_wape": naive["mean_wape"], "wape_by_node": naive["mean_wape_by_node"], "n_features": 0}
    print(f"  {'naive (1h persistence)':32s} WAPE {naive['mean_wape']:6.2f}%  Puerto Montt {naive['mean_wape_by_node']['Puerto_Montt_220kV']:6.2f}%")
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    run()
