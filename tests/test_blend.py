"""Mezcla por barra entre el modelo de precio y la persistencia."""
import numpy as np
import pandas as pd

from src.models.blend import apply_blend, blend_weights


def _panel(n_hours: int = 600, seed: int = 0) -> pd.DataFrame:
    """Dos barras: en "signal" el precio sigue a una variable x; en "sticky" es
    un paseo aleatorio que x no explica, donde lo mejor es el precio anterior."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n_hours, freq="h")
    x = rng.normal(size=n_hours)
    signal = 50 + 10 * x
    sticky = 50 + np.cumsum(rng.normal(size=n_hours))
    rows = []
    for node, y in (("signal", signal), ("sticky", sticky)):
        lag = np.r_[y[0], y[:-1]]
        rows.append(pd.DataFrame({"timestamp": ts, "node": node, "x": x, "y": y, "y_lag_1h": lag}))
    return pd.concat(rows, ignore_index=True)


def test_blend_weights_trust_the_model_where_it_explains_the_price_and_persistence_where_it_does_not():
    df = _panel()
    params = {"n_estimators": 60, "learning_rate": 0.2, "num_leaves": 15, "random_state": 0, "verbosity": -1}
    w = blend_weights(df, ["x", "y_lag_1h"], "y", params, [], "y_lag_1h")
    assert w["signal"] >= 0.8
    assert w["sticky"] <= 0.3


def test_apply_blend_uses_each_rows_node_weight_and_keeps_unknown_nodes():
    out = apply_blend(np.array([10.0, 10.0, 10.0]), ["a", "b", "c"], np.array([0.0, 0.0, 0.0]), {"a": 1.0, "b": 0.25})
    assert out.tolist() == [10.0, 2.5, 10.0]
