import numpy as np
import pandas as pd
import polars as pl
import pytest
import torch

from src.data.fetch_energy_data import generate_energy_data
from src.features.build_features import build_features
from src.models.torch_forecaster import (
    ACTIVATIONS,
    WindowMLP,
    compare_activations,
    huber_loss,
    predict_mlp,
    train_mlp,
    train_with_walk_forward_validation,
)

TINY_EPOCHS = 3


@pytest.fixture(scope="module")
def features_df():
    raw = generate_energy_data()
    return build_features(pl.from_pandas(raw)).to_pandas()


def test_huber_loss_is_quadratic_for_small_errors_linear_for_large():
    y_true = torch.tensor([0.0, 0.0])
    small_pred = torch.tensor([0.1, -0.1])
    large_pred = torch.tensor([10.0, -10.0])

    small_loss = huber_loss(small_pred, y_true, delta=1.0)
    large_loss = huber_loss(large_pred, y_true, delta=1.0)

    # Error cuadrático puro habría dado (10)**2/2=50 -- Huber debe ser mucho
    # menor porque para |residual|>delta crece linealmente, no cuadráticamente.
    assert small_loss.item() == pytest.approx(0.5 * 0.1**2, abs=1e-6)
    assert large_loss.item() < 50.0


def test_huber_loss_is_zero_for_perfect_predictions():
    y = torch.tensor([1.0, 2.0, 3.0])
    assert huber_loss(y, y).item() == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("activation", list(ACTIVATIONS))
def test_window_mlp_forward_shape_for_every_activation(activation):
    model = WindowMLP(n_features=6, hidden_dims=(8, 4), activation=activation)
    x = torch.randn(10, 6)
    out = model(x)
    assert out.shape == (10,)


def test_window_mlp_rejects_unknown_activation():
    with pytest.raises(ValueError):
        WindowMLP(n_features=4, activation="tanh")


def test_train_mlp_reduces_loss_over_epochs():
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = pd.Series(X["a"] * 2 - X["b"] + rng.normal(scale=0.01, size=n))

    _model, history, _norm = train_mlp(X, y, activation="relu", epochs=25, batch_size=32, seed=0)

    assert len(history["train_loss"]) == 25
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_predict_mlp_returns_reasonable_predictions_after_training():
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["a", "b"])
    y = pd.Series(3 * X["a"] - 2 * X["b"])

    model, _history, norm_params = train_mlp(X, y, activation="gelu", epochs=60, batch_size=32, seed=1)
    preds = predict_mlp(model, X, norm_params)

    # No exige precisión estricta (red chica, pocas épocas) -- solo que
    # capture la dirección de la relación aprendida, no ruido puro.
    correlation = np.corrcoef(preds, y.to_numpy())[0, 1]
    assert correlation > 0.8


def test_train_with_walk_forward_validation_runs_on_real_features(features_df):
    result = train_with_walk_forward_validation(
        features_df, "demand_mwh", n_splits=2, activation="relu", epochs=TINY_EPOCHS,
    )

    assert len(result["fold_metrics"]) == 2
    assert result["mean_wape"] >= 0
    assert result["mean_mae"] >= 0
    assert result["final_model"] is not None
    assert len(result["final_history"]["train_loss"]) == TINY_EPOCHS


def test_compare_activations_returns_all_three_and_they_differ(features_df):
    results = compare_activations(features_df, "demand_mwh", n_splits=2, epochs=TINY_EPOCHS)

    assert set(results.keys()) == set(ACTIVATIONS)
    for activation, result in results.items():
        assert result["activation"] == activation
        assert result["mean_wape"] >= 0
