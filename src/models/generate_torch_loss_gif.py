"""Genera el GIF animado del loss/época del MLP PyTorch (Swish, la mejor
activación por WAPE promedio segun `generate_torch_report`), reentrenando
sobre los mismos folds walk-forward para obtener la historia real de loss
por época -- ningún valor fabricado. No reentrena LightGBM/XGBoost ni
persiste en DuckDB (eso ya lo hace `generate_torch_report.py`); este script
solo agrega el artefacto GIF junto al PNG ya existente en `data/processed/`.

Ejecutar con: `python -m src.models.generate_torch_loss_gif`
"""
from __future__ import annotations

from src.models.torch_forecaster import DEFAULT_TARGET, N_SPLITS, train_with_walk_forward_validation
from src.models.torch_plots import animate_loss_curve
from src.models.train_forecaster import load_features

BEST_ACTIVATION = "swish"


def run(target_column: str = DEFAULT_TARGET, n_splits: int = N_SPLITS, epochs: int = 40) -> None:
    df = load_features()
    print(f"Entrenando MLP PyTorch ({BEST_ACTIVATION}) -- {target_column} para la historia de loss real...")
    result = train_with_walk_forward_validation(
        df, target_column, n_splits=n_splits, activation=BEST_ACTIVATION, epochs=epochs,
    )
    print(f"WAPE promedio: {result['mean_wape']:.2f}%")
    out_path = animate_loss_curve(result["final_history"], target_column, BEST_ACTIVATION)
    print(f"GIF guardado en {out_path}")


if __name__ == "__main__":
    run()
