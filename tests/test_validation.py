import numpy as np
import pandas as pd
import pytest

from src.models.validation import get_walk_forward_folds, mean_by_node, wape, wape_by_node


def test_wape_perfect_prediction_is_zero():
    y = np.array([10.0, 20.0, 30.0])
    assert wape(y, y) == 0.0


def test_wape_known_value():
    y_true = np.array([10.0, 10.0, 10.0, 10.0])
    y_pred = np.array([12.0, 8.0, 10.0, 10.0])
    # sum(|error|) = 4, sum(|actual|) = 40 -> 10%
    assert wape(y_true, y_pred) == pytest.approx(10.0)


def test_wape_all_zero_actuals_is_nan():
    y = np.zeros(5)
    assert np.isnan(wape(y, y + 1))


def test_folds_are_chronologically_disjoint():
    timestamps = pd.date_range("2024-01-01", periods=200, freq="h")
    df = pd.DataFrame({"timestamp": timestamps})

    folds = get_walk_forward_folds(df, n_splits=4)
    assert len(folds) == 4
    for train_ts, test_ts in folds:
        assert train_ts.isdisjoint(test_ts)
        assert max(train_ts) < min(test_ts)


def test_folds_grow_across_splits():
    timestamps = pd.date_range("2024-01-01", periods=200, freq="h")
    df = pd.DataFrame({"timestamp": timestamps})

    folds = get_walk_forward_folds(df, n_splits=4)
    train_sizes = [len(train_ts) for train_ts, _ in folds]
    assert train_sizes == sorted(train_sizes)


def test_wape_by_node_splits_the_error_per_bus():
    per_node = wape_by_node(["a", "a", "b", "b"], [10, 10, 100, 100], [11, 9, 100, 50])
    assert per_node == {"a": 10.0, "b": 25.0}
    folds = [{"wape_by_node": {"a": 10.0, "b": 20.0}}, {"wape_by_node": {"a": 30.0, "b": 40.0}}]
    assert mean_by_node(folds) == {"a": 20.0, "b": 30.0}
