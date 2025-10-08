"""Utility helpers for running the full ExoStack workflow in Google Colab.

This module keeps the notebook-style progress reporting that the original
script provided, while delegating the heavy lifting to the reusable pipeline
functions contained in :mod:`exostack.pipeline`.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

from .pipeline import (
    ModelMetrics,
    TrainingResults,
    load_kepler_cumulative_table,
    predict_exoplanet_ultimate,
    train_ultimate_model,
)

DATA_URL = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&format=csv"

EXAMPLE_SCENARIOS = [
    {
        "name": "Earth-like in Habitable Zone",
        "emoji": "🌍",
        "params": {
            "orbital_period": 365.25,
            "transit_duration": 13.0,
            "transit_depth": 84.0,
            "planet_radius": 1.0,
            "equilibrium_temp": 288,
            "insolation_flux": 1.0,
            "stellar_temp": 5778,
            "stellar_radius": 1.0,
            "stellar_mass": 1.0,
            "model_snr": 25.0,
        },
    },
    {
        "name": "Hot Jupiter",
        "emoji": "🔥",
        "params": {
            "orbital_period": 3.5,
            "transit_duration": 4.0,
            "transit_depth": 15000.0,
            "planet_radius": 11.2,
            "equilibrium_temp": 1500.0,
            "insolation_flux": 150.0,
            "stellar_temp": 6000.0,
            "stellar_radius": 1.2,
            "stellar_mass": 1.1,
            "model_snr": 45.0,
        },
    },
    {
        "name": "Super-Earth in Habitable Zone",
        "emoji": "🌏",
        "params": {
            "orbital_period": 37.5,
            "transit_duration": 8.5,
            "transit_depth": 450.0,
            "planet_radius": 1.6,
            "equilibrium_temp": 265.0,
            "insolation_flux": 0.7,
            "stellar_temp": 5200.0,
            "stellar_radius": 0.9,
            "stellar_mass": 0.85,
            "model_snr": 30.0,
        },
    },
    {
        "name": "Mini-Neptune",
        "emoji": "💙",
        "params": {
            "orbital_period": 12.8,
            "transit_duration": 5.2,
            "transit_depth": 2800.0,
            "planet_radius": 3.2,
            "equilibrium_temp": 620.0,
            "insolation_flux": 8.5,
            "stellar_temp": 5500.0,
            "stellar_radius": 1.05,
            "stellar_mass": 1.0,
            "model_snr": 35.0,
        },
    },
]


def _print_banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _summarize_metrics(name: str, metrics: ModelMetrics) -> str:
    return (
        f"{name:20s} | Acc: {metrics.accuracy * 100:.2f}% | "
        f"F1: {metrics.f1 * 100:.2f}% | AUC: {metrics.auc:.4f}"
    )


def _package_training_results(results: TrainingResults) -> Dict[str, object]:
    """Convert :class:`TrainingResults` to a serializable dictionary."""

    base_metrics = {name: asdict(metric) for name, metric in results.base_model_metrics.items()}

    return {
        "version": "3.1-colab",
        "created": datetime.utcnow().isoformat(),
        "artifacts": results.artifacts,
        "class_distribution": results.class_distribution,
        "feature_count": results.feature_count,
        "selected_features": results.selected_features,
        "metrics": {
            "stacking": asdict(results.stacking_metrics),
            "base": base_metrics,
        },
    }


def run_colab_workflow(
    *,
    data_url: str = DATA_URL,
    model_package_path: str = "exostack_detector.pkl",
    example_scenarios: Iterable[Dict[str, object]] = EXAMPLE_SCENARIOS,
) -> TrainingResults:
    """Execute the full training pipeline with notebook-style logging."""

    print("🚀 NASA EXOPLANET DETECTION - EXOSTACK EDITION")
    print("=" * 70)
    print("Target: 95%+ Accuracy with Hyperparameter Optimization")
    print("=" * 70)

    _print_banner("🌌 Loading REAL NASA Kepler Dataset")
    df = load_kepler_cumulative_table(data_url)
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    _print_banner("🤖 Training ultimate stacking ensemble")
    results = train_ultimate_model(df)
    print("✅ Training complete!")

    stacking = results.stacking_metrics
    print(
        f"\n✨ STACKING ENSEMBLE PERFORMANCE:\n"
        f"   Accuracy : {stacking.accuracy * 100:.2f}%\n"
        f"   Precision: {stacking.precision * 100:.2f}%\n"
        f"   Recall   : {stacking.recall * 100:.2f}%\n"
        f"   F1-Score : {stacking.f1 * 100:.2f}%\n"
        f"   AUC-ROC  : {stacking.auc:.4f}"
    )

    _print_banner("🏆 TOP 4 BASE MODELS")
    top_models = sorted(
        results.base_model_metrics.items(), key=lambda item: item[1].accuracy, reverse=True
    )[:4]
    for rank, (name, metrics) in enumerate(top_models, 1):
        print(f"{rank}. {_summarize_metrics(name, metrics)}")

    _print_banner("🌟 EXAMPLE PREDICTIONS")
    for scenario in example_scenarios:
        print("\n" + "-" * 70)
        print(f"{scenario['emoji']}  {scenario['name']}")
        detailed_prediction = predict_exoplanet_ultimate(
            results.artifacts,
            detailed=True,
            **scenario["params"],
        )
        print(detailed_prediction["interpretation"])
        print(
            f"📊 CONFIRMED: {detailed_prediction['confirmed_probability'] * 100:.2f}% | "
            f"NOT CONFIRMED: {detailed_prediction['non_confirmed_probability'] * 100:.2f}%"
        )

    _print_banner("💾 Saving model package")
    package = _package_training_results(results)
    path = Path(model_package_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(package, fh)
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"✅ Saved '{path}' ({size_mb:.2f} MB)")

    _print_banner("🎉 Workflow complete")

    return results


if __name__ == "__main__":  # pragma: no cover - manual entry point
    run_colab_workflow()
