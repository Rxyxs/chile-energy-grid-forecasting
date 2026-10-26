"""Tercer enfoque de modelado: un MLP en PyTorch sobre las mismas features de
ventana temporal (lags/rolling) que ya usan LightGBM (`train_forecaster.py`) y
XGBoost (`rolling_stability.py`) -- mismos folds walk-forward, misma métrica
(WAPE/MAE), para que la comparación de 3 familias de modelos en el README sea
directa. No es una arquitectura secuencial (LSTM/GRU): las features ya son
tabulares (ventanas de lags calculadas en `build_features.py`), así que un MLP
denso sobre ese mismo input tabular es la comparación justa "árbol vs. red
neuronal" sobre el mismo feature set, en vez de introducir una arquitectura
recurrente que necesitaría su propio pipeline de secuencias.

Dos cosas que este módulo agrega y los otros dos enfoques no tenían:
- Una loss custom (Huber/SmoothL1), más robusta a outliers que el MSE
  implícito de un `LGBMRegressor`/`XGBRegressor` por defecto -- relevante acá
  porque `marginal_cost_usd_mwh` tiene spikes de estrés de oferta que un MSE
  penalizaría (y por lo tanto ajustaría) desproporcionadamente.
- Una comparación controlada de función de activación (ReLU/GELU/SiLU=Swish)
  manteniendo arquitectura, loss, optimizador y seed fijos -- para aislar el
  efecto de la no-linealidad en sí, no confundirlo con otra diferencia.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.real_panel import rows_for_target
from src.models.train_forecaster import CATEGORICAL_FEATURES, TARGET_COLUMNS, get_feature_columns, load_features
from src.models.validation import get_walk_forward_folds, wape

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TORCH_METRICS_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "torch_forecaster_metrics.json"

DEFAULT_TARGET = "marginal_cost_usd_mwh"
N_SPLITS = 5
HIDDEN_DIMS = (64, 32)
MIN_EPOCHS = 100   # ninguna red se detiene antes de 100 épocas
MAX_EPOCHS = 300
PATIENCE = 20     # épocas sin mejora en validación antes de cortar (solo después del mínimo)
VAL_FRACTION = 0.15
LR_FACTOR = 0.5
LR_PATIENCE = 8
DEFAULT_EPOCHS = MAX_EPOCHS
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "checkpoints"
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
HUBER_DELTA = 1.0

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "swish": nn.SiLU,  # SiLU(x) = x * sigmoid(x) == Swish
}


class WindowMLP(nn.Module):
    """MLP simple sobre un vector de features de ventana ya calculado
    (lags/rolling de `build_features.py`), con la función de activación
    inyectable para poder comparar ReLU/GELU/Swish manteniendo el resto de
    la arquitectura idéntica."""

    def __init__(self, n_features: int, hidden_dims: tuple[int, ...] = HIDDEN_DIMS, activation: str = "relu"):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"Activación desconocida: {activation!r}. Opciones: {sorted(ACTIVATIONS)}")
        act_cls = ACTIVATIONS[activation]

        layers: list[nn.Module] = []
        in_dim = n_features
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(in_dim, hidden_dim), act_cls()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def huber_loss(y_pred: torch.Tensor, y_true: torch.Tensor, delta: float = HUBER_DELTA) -> torch.Tensor:
    """Huber/SmoothL1 manual (en vez de `nn.SmoothL1Loss` directo) para dejar
    explícito el tradeoff que motiva usarla acá: cuadrática (como MSE) para
    errores chicos, lineal para errores grandes -- así un spike de precio no
    domina el gradiente del batch como lo haría con MSE puro."""
    residual = y_true - y_pred
    abs_residual = residual.abs()
    quadratic = torch.clamp(abs_residual, max=delta)
    linear = abs_residual - quadratic
    return (0.5 * quadratic.pow(2) + delta * linear).mean()


def _encode_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """One-hot encode de las features categóricas (`node`) -- a diferencia de
    LightGBM/XGBoost, que consumen categóricas nativamente, un MLP necesita
    input puramente numérico."""
    encoded = pd.get_dummies(df[feature_columns], columns=CATEGORICAL_FEATURES)
    return encoded.astype("float32")


def _to_tensor(df: pd.DataFrame) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(df.to_numpy(dtype=np.float32)))


def _fingerprint(*frames, **config) -> str:
    """Huella de los datos y la configuración: un checkpoint solo se retoma si coincide."""
    h = hashlib.sha256()
    for f in frames:
        if f is not None:
            h.update(np.ascontiguousarray(np.asarray(f, dtype=np.float32)).tobytes())
    h.update(json.dumps(config, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def train_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    activation: str = "relu",
    epochs: int = MAX_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    seed: int = 42,
    min_epochs: int = MIN_EPOCHS,
    patience: int = PATIENCE,
    checkpoint_path: Path | None = None,
) -> tuple[WindowMLP, dict, dict]:
    """Entrena un `WindowMLP` con Adam + Huber loss, estandarizando X/y con
    estadísticas de train únicamente (evita fuga de información del set de
    validación). Devuelve el modelo, la historia de loss por época (para el
    gráfico loss/epoch) y los parámetros de normalización necesarios para
    predecir sobre datos nuevos con `predict_mlp`.

    Con validación: entrena al menos `min_epochs` épocas (o `epochs`, si es
    menor) y desde ahí early stopping corta cuando la loss de validación no
    mejora en `patience` épocas; se restauran los pesos de la mejor época y
    la tasa de aprendizaje baja a la mitad cuando la validación se estanca.
    Sin validación entrena `epochs` épocas completas.

    Con `checkpoint_path`, el estado completo (pesos, optimizador, scheduler,
    historia, mejor modelo) se guarda cada época con escritura atómica; si el
    proceso se corta, la siguiente llamada con los mismos datos y
    configuración retoma desde la última época guardada.
    """
    torch.manual_seed(seed)

    x_mean, x_std = X_train.mean(), X_train.std().replace(0, 1.0)
    y_mean, y_std = float(y_train.mean()), float(y_train.std() or 1.0)

    X_train_t = _to_tensor((X_train - x_mean) / x_std)
    y_train_t = torch.from_numpy(((y_train - y_mean) / y_std).to_numpy(dtype=np.float32))

    model = WindowMLP(n_features=X_train.shape[1], activation=activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator()
    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True, generator=generator)

    has_val = X_val is not None and y_val is not None and len(X_val) > 0
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=LR_FACTOR, patience=LR_PATIENCE)
                 if has_val else None)
    if has_val:
        X_val_t = _to_tensor((X_val - x_mean) / x_std)
        y_val_t = torch.from_numpy(((y_val - y_mean) / y_std).to_numpy(dtype=np.float32))

    fp = _fingerprint(X_train, y_train, X_val if has_val else None, y_val if has_val else None,
                      activation=activation, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed,
                      min_epochs=min_epochs, patience=patience, hidden=HIDDEN_DIMS)
    state = {"epoch": 0, "train_loss": [], "val_loss": [], "best_val": float("inf"), "best_epoch": 0,
             "best_state": None, "bad_epochs": 0, "done": False}
    resumed_from = 0
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if ckpt.get("fingerprint") == fp:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler is not None and ckpt.get("scheduler") is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
            state = ckpt["state"]
            resumed_from = state["epoch"]

    def save_checkpoint() -> None:
        if checkpoint_path is None:
            return
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({"fingerprint": fp, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None, "state": state}, tmp)
        tmp.replace(path)  # un corte a mitad de escritura no deja un checkpoint corrupto

    while not state["done"] and state["epoch"] < epochs:
        generator.manual_seed(seed + state["epoch"])  # mismo orden de batches al retomar
        model.train()
        epoch_losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = huber_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        state["train_loss"].append(float(np.mean(epoch_losses)))
        state["epoch"] += 1

        if has_val:
            model.eval()
            with torch.no_grad():
                val_loss = float(huber_loss(model(X_val_t), y_val_t).item())
            state["val_loss"].append(val_loss)
            scheduler.step(val_loss)
            if val_loss < state["best_val"] - 1e-6:
                state.update(best_val=val_loss, best_epoch=state["epoch"], bad_epochs=0,
                             best_state=copy.deepcopy(model.state_dict()))
            else:
                state["bad_epochs"] += 1
            if state["epoch"] >= min(min_epochs, epochs) and state["bad_epochs"] >= patience:
                state["done"] = True
        save_checkpoint()

    if has_val and state["best_state"] is not None:
        model.load_state_dict(state["best_state"])
    state["done"] = True
    save_checkpoint()

    history = {
        "train_loss": state["train_loss"],
        "val_loss": state["val_loss"],
        "epochs_run": state["epoch"],
        "best_epoch": state["best_epoch"] if has_val else state["epoch"],
        "stopped_early": bool(has_val and state["epoch"] < epochs),
        "resumed_from": resumed_from,
    }
    norm_params = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    return model, history, norm_params


def predict_mlp(model: WindowMLP, X: pd.DataFrame, norm_params: dict) -> np.ndarray:
    model.eval()
    X_norm = (X - norm_params["x_mean"]) / norm_params["x_std"]
    with torch.no_grad():
        y_norm = model(_to_tensor(X_norm)).numpy()
    return y_norm * norm_params["y_std"] + norm_params["y_mean"]


def train_with_walk_forward_validation(
    df: pd.DataFrame,
    target_column: str = DEFAULT_TARGET,
    n_splits: int = N_SPLITS,
    activation: str = "relu",
    epochs: int = MAX_EPOCHS,
    seed: int = 42,
    min_epochs: int = MIN_EPOCHS,
    patience: int = PATIENCE,
    checkpoint_dir: Path | None = None,
) -> dict:
    """Análogo directo a `train_forecaster.train_with_walk_forward_validation`,
    mismos folds walk-forward, pero con el `WindowMLP` de PyTorch. El modelo
    final devuelto es el del último fold, igual criterio que LightGBM.

    Dentro de cada fold, el último `VAL_FRACTION` de las horas de train
    (todos los nodos a la vez) se separa como validación para el early
    stopping; las horas de test del fold no se usan hasta la evaluación.
    """
    feature_columns = get_feature_columns(df)
    clean_df = rows_for_target(df, target_column).dropna(subset=feature_columns + [target_column]).reset_index(drop=True)
    encoded = _encode_features(clean_df, feature_columns)

    fold_metrics = []
    last_model, last_history, last_norm = None, None, None
    for fold, (train_ts, test_ts) in enumerate(get_walk_forward_folds(clean_df, n_splits)):
        ordered = np.sort(np.array(list(train_ts)))
        n_val = max(1, int(len(ordered) * VAL_FRACTION))
        val_ts = set(ordered[-n_val:])
        fit_mask = clean_df["timestamp"].isin(train_ts) & ~clean_df["timestamp"].isin(val_ts)
        val_mask = clean_df["timestamp"].isin(val_ts)
        test_mask = clean_df["timestamp"].isin(test_ts)

        X_fit, y_fit = encoded.loc[fit_mask], clean_df.loc[fit_mask, target_column]
        X_val, y_val = encoded.loc[val_mask], clean_df.loc[val_mask, target_column]
        X_test, y_test = encoded.loc[test_mask], clean_df.loc[test_mask, target_column]

        ckpt = None if checkpoint_dir is None else Path(checkpoint_dir) / f"{target_column}_{activation}_fold{fold}.ckpt"
        model, history, norm_params = train_mlp(
            X_fit, y_fit, X_val, y_val, activation=activation, epochs=epochs, seed=seed,
            min_epochs=min_epochs, patience=patience, checkpoint_path=ckpt,
        )
        y_pred = predict_mlp(model, X_test, norm_params)

        fold_metrics.append({
            "fold": fold,
            "train_rows": int(len(X_fit)),
            "val_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "epochs_run": history["epochs_run"],
            "best_epoch": history["best_epoch"],
            "best_val_loss": float(min(history["val_loss"])) if history["val_loss"] else float("nan"),
            "wape": wape(y_test.to_numpy(), y_pred),
            "mae": float(np.abs(y_test.to_numpy() - y_pred).mean()),
        })
        last_model, last_history, last_norm = model, history, norm_params

    return {
        "target_column": target_column,
        "activation": activation,
        "n_splits": n_splits,
        "max_epochs": epochs,
        "min_epochs": min_epochs,
        "patience": patience,
        "fold_metrics": fold_metrics,
        "mean_wape": float(np.mean([m["wape"] for m in fold_metrics])),
        "mean_mae": float(np.mean([m["mae"] for m in fold_metrics])),
        "mean_best_val_loss": float(np.mean([m["best_val_loss"] for m in fold_metrics])),
        "final_model": last_model,
        "final_history": last_history,
        "final_norm_params": last_norm,
        "feature_columns": list(encoded.columns),
    }


def compare_activations(
    df: pd.DataFrame,
    target_column: str = DEFAULT_TARGET,
    n_splits: int = N_SPLITS,
    epochs: int = DEFAULT_EPOCHS,
    seed: int = 42,
    checkpoint_dir: Path | None = CHECKPOINT_DIR,
) -> dict:
    """Corre `train_with_walk_forward_validation` una vez por activación
    (ReLU/GELU/Swish), manteniendo arquitectura/loss/optimizador/seed fijos,
    para que la única variable que cambia entre corridas sea la no-linealidad."""
    results = {}
    for activation in ACTIVATIONS:
        result = train_with_walk_forward_validation(
            df, target_column, n_splits=n_splits, activation=activation, epochs=epochs, seed=seed,
            checkpoint_dir=checkpoint_dir,
        )
        results[activation] = {k: v for k, v in result.items() if k != "final_model"}
        results[activation]["final_model"] = result["final_model"]
    return results


def run_full_pipeline(n_splits: int = N_SPLITS, epochs: int = DEFAULT_EPOCHS) -> dict:
    """Corre la comparación de activaciones para los 4 targets, guarda las
    métricas (sin los objetos de modelo, no serializables a JSON) y devuelve
    el dict completo (con modelos) para que quien llame pueda además graficar."""
    df = load_features()
    all_metrics: dict[str, dict] = {}

    for target in TARGET_COLUMNS:
        print(f"\n=== MLP (PyTorch) -- {target} ===")
        activation_results = compare_activations(df, target, n_splits=n_splits, epochs=epochs)
        for activation, result in activation_results.items():
            print(f"  {activation:6s} -> WAPE promedio: {result['mean_wape']:.2f}% | MAE promedio: {result['mean_mae']:.2f}")
        all_metrics[target] = activation_results

    serializable = {
        target: {
            activation: {k: v for k, v in result.items() if k not in ("final_model", "final_norm_params")}
            for activation, result in activations.items()
        }
        for target, activations in all_metrics.items()
    }
    TORCH_METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TORCH_METRICS_OUTPUT_PATH.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"\nMétricas guardadas en {TORCH_METRICS_OUTPUT_PATH}")
    return all_metrics


if __name__ == "__main__":
    run_full_pipeline()
