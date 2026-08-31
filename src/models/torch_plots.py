"""Gráficos explicativos del MLP PyTorch, en el mismo estilo/paleta que los
gráficos matplotlib ya existentes en `notebooks/02_Weather_Augmented_Rolling_CV.ipynb`
(`#1565c0` azul, `#9aa0a6` gris, `#e05252` rojo, `plt.rcParams["figure.dpi"] = 110`),
guardados en `data/processed/` junto a `weather_ablation_*.png` y
`rolling_origin_stability.png` -- misma carpeta de outputs, no una nueva.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.dpi"] = 110

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

COLOR_ACTUAL = "#1565c0"
COLOR_PRED = "#e05252"
COLOR_GRAY = "#9aa0a6"
ACTIVATION_COLORS = {"relu": "#1565c0", "gelu": "#1baf7a", "swish": "#e05252"}


def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, target_column: str, out_path: Path | None = None) -> Path:
    out_path = out_path or OUTPUT_DIR / f"torch_pred_vs_actual_{target_column}.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=10, alpha=0.4, color=COLOR_ACTUAL)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, color=COLOR_GRAY, linestyle="--", linewidth=1, label="y = x")
    ax.set_xlabel("Real")
    ax.set_ylabel("Predicho (MLP PyTorch)")
    ax.set_title(f"Predicho vs. real -- {target_column}")
    ax.legend()
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, target_column: str, out_path: Path | None = None) -> Path:
    out_path = out_path or OUTPUT_DIR / f"torch_residuals_{target_column}.png"
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(residuals, bins=60, color=COLOR_ACTUAL, alpha=0.8)
    ax.axvline(0, color=COLOR_GRAY, linestyle="--", linewidth=1)
    ax.set_xlabel(f"Residual (real - predicho), std={residuals.std():.2f}")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Residuos -- {target_column}")
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_loss_curve(history: dict, target_column: str, activation: str, out_path: Path | None = None) -> Path:
    out_path = out_path or OUTPUT_DIR / f"torch_loss_curve_{target_column}.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["train_loss"], color=COLOR_ACTUAL, label="Train (Huber)")
    if history.get("val_loss"):
        ax.plot(history["val_loss"], color=COLOR_PRED, label="Val (Huber)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss (Huber, y estandarizado)")
    ax.set_title(f"Loss por época -- {target_column} ({activation})")
    ax.legend()
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_activation_comparison(activation_results: dict, target_column: str, out_path: Path | None = None) -> Path:
    """Barra comparativa de WAPE promedio por activación (ReLU/GELU/Swish),
    arquitectura/loss/seed fijos -- ver `torch_forecaster.compare_activations`."""
    out_path = out_path or OUTPUT_DIR / f"torch_activation_comparison_{target_column}.png"
    activations = list(activation_results.keys())
    wapes = [activation_results[a]["mean_wape"] for a in activations]
    colors = [ACTIVATION_COLORS.get(a, COLOR_GRAY) for a in activations]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(activations, wapes, color=colors)
    ax.set_ylabel("WAPE walk-forward promedio (%)")
    ax.set_title(f"ReLU vs. GELU vs. Swish -- {target_column}")
    for i, w in enumerate(wapes):
        ax.text(i, w, f"{w:.2f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
