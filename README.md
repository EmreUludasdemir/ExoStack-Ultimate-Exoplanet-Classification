# ExoStack: Ultimate Exoplanet Classification

ExoStack is a modular Python implementation of the "ultra advanced" exoplanet
classification workflow shared in the original notebook-style script.  The code
has been restructured into reusable functions that can be executed easily inside
Google Colab or any local Python environment.

## 🚀 Google Colab Quickstart

```python
# 1. Install the scientific stack (run in a fresh Colab cell)
!pip install -q pandas numpy scikit-learn matplotlib seaborn
!pip install -q imbalanced-learn xgboost lightgbm catboost

# 2. Clone the repository
!git clone https://github.com/<your-account>/ExoStack-Ultimate-Exoplanet-Classification.git
%cd ExoStack-Ultimate-Exoplanet-Classification

# 3. Run the full training workflow with notebook-style logs
from exostack import run_colab_workflow
results = run_colab_workflow()
```

The helper returns a `TrainingResults` dataclass containing trained estimators,
metrics, and metadata.  A serialized model bundle is written to
`exostack_detector.pkl` for later reuse.

## 🧰 Core Modules

| Module | Description |
| --- | --- |
| `exostack.pipeline` | Preprocessing, feature engineering, model training, and prediction helpers. |
| `exostack.colab_workflow` | End-to-end runner that mirrors the verbose output of the original notebook. |
| `exostack.__init__` | Convenience imports for the public API. |

### Key Functions

- `load_kepler_cumulative_table(url)` – Downloads the Kepler cumulative table
  from NASA.
- `train_ultimate_model(df)` – Trains the SMOTEENN-balanced ensemble, returning
  a `TrainingResults` object with metrics and fitted estimators.
- `predict_exoplanet_ultimate(artifacts, ..., detailed=True)` – Predicts whether
  a candidate is a confirmed exoplanet.  When `detailed=True` it also returns an
  interpretation block similar to the notebook output.
- `run_colab_workflow()` – Executes the entire pipeline, prints progress,
  showcases example predictions, and stores the serialized package.

## 📦 Saved Package Contents

`run_colab_workflow()` generates a `pickle` file that stores:

- Serialized preprocessing artifacts (imputer, feature selector, scaler).
- The stacking ensemble alongside the top-performing base models.
- Cross-validation metrics for every model.
- Train/test class distributions and the selected feature names.

## 🧪 Example: Manual Usage

```python
from exostack import (
    load_kepler_cumulative_table,
    train_ultimate_model,
    predict_exoplanet_ultimate,
)

url = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&format=csv"
df = load_kepler_cumulative_table(url)
results = train_ultimate_model(df)

prediction = predict_exoplanet_ultimate(
    results.artifacts,
    orbital_period=3.5,
    transit_duration=4.0,
    transit_depth=15000.0,
    planet_radius=11.2,
    equilibrium_temp=1500.0,
    insolation_flux=150.0,
    stellar_temp=6000.0,
    stellar_radius=1.2,
    stellar_mass=1.1,
    model_snr=45.0,
    detailed=True,
)

print(prediction["interpretation"])
```

## 📚 Scientific References

- Luz et al., 2024 – Ensemble-based exoplanet classification (Electronics).
- Malik et al., 2022 – Advanced feature engineering for Kepler/TESS candidates
  (MNRAS).

## 🪐 License

[MIT](LICENSE)
