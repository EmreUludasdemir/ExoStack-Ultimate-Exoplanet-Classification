"""Public API for the ExoStack training utilities."""

from .colab_workflow import run_colab_workflow
from .pipeline import (
    CRITICAL_FEATURES,
    IDENTIFIER_COLUMNS,
    ModelArtifacts,
    ModelMetrics,
    TrainingResults,
    load_kepler_cumulative_table,
    predict_exoplanet_ultimate,
    train_ultimate_model,
)

__all__ = [
    "CRITICAL_FEATURES",
    "IDENTIFIER_COLUMNS",
    "ModelArtifacts",
    "ModelMetrics",
    "TrainingResults",
    "load_kepler_cumulative_table",
    "predict_exoplanet_ultimate",
    "train_ultimate_model",
    "run_colab_workflow",
]
