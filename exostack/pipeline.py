"""Training pipeline for the ExoStack exoplanet classification system.

This module encapsulates the full training routine that is described in the
project README.  The implementation closely follows the notebook-style script
shared by the user, but it has been adapted into testable, reusable Python
functions.  The focus of this rewrite is to make the preprocessing steps
robust so that feature selection and model training no longer fail because of
NaN values that appear after feature engineering.

The pipeline exposes two public helper functions:

``train_ultimate_model``
    Downloads (or accepts) the Kepler cumulative table, performs feature
    engineering, fits the base models, the stacking ensemble and returns a
    ``TrainingResults`` data object containing trained estimators and useful
    metadata.

``predict_exoplanet_ultimate``
    Mirrors the heavy feature engineering used during training and produces a
    classification for a single candidate when supplied with the trained
    artifacts.

The code is intentionally verbose so that it is easy to follow what happens at
each step.  This makes future debugging significantly easier than with a single
monolithic notebook cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

CRITICAL_FEATURES: Tuple[str, ...] = (
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_impact",
    "koi_steff",
    "koi_srad",
    "koi_smass",
    "koi_slogg",
    "koi_model_snr",
    "koi_tce_plnt_num",
)

IDENTIFIER_COLUMNS: Tuple[str, ...] = (
    "rowid",
    "kepid",
    "kepoi_name",
    "kepler_name",
    "koi_pdisposition",
    "koi_score",
)


@dataclass
class ModelMetrics:
    """Container that stores evaluation metrics for a single estimator."""

    cv_mean: float
    cv_std: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float


@dataclass
class ModelArtifacts:
    """Artifacts produced during training that are required for inference."""

    feature_columns: List[str]
    imputer: SimpleImputer
    selector: SelectKBest
    scaler: RobustScaler
    selected_feature_names: List[str]
    trained_models: Dict[str, object]
    stacking_model: StackingClassifier


@dataclass
class TrainingResults:
    """Aggregate object returned by :func:`train_ultimate_model`."""

    artifacts: ModelArtifacts
    base_model_metrics: Dict[str, ModelMetrics]
    stacking_metrics: ModelMetrics
    class_distribution: Dict[str, Dict[str, int]]
    feature_count: int
    X_test: np.ndarray
    y_test: np.ndarray
    selected_features: List[str]


def load_kepler_cumulative_table(url: str) -> pd.DataFrame:
    """Load the Kepler cumulative table from the NASA archive."""

    return pd.read_csv(url, comment="#")


def _remove_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [col for col in IDENTIFIER_COLUMNS if col in df.columns]
    return df.drop(columns=columns_to_drop, errors="ignore")


def _fill_base_feature_gaps(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    """Fill missing values on the base Kepler columns using the median."""

    df_filled = df.copy()
    for feature in features:
        if feature in df_filled.columns:
            median_value = df_filled[feature].median()
            df_filled[feature] = df_filled[feature].fillna(median_value)
    return df_filled


def _remove_outliers(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    """Remove extreme outliers using a 3-sigma rule applied feature-wise."""

    if df.empty:
        return df

    mask = np.ones(len(df), dtype=bool)
    for feature in features:
        if feature not in df.columns:
            continue
        series = df[feature]
        mean = series.mean()
        std = series.std()
        if not np.isfinite(std) or std == 0:
            continue
        feature_mask = (series - mean).abs() <= 3 * std
        mask &= feature_mask.fillna(False)
    return df.loc[mask]


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the engineered feature set used throughout the project."""

    engineered = df.copy()

    if {"koi_prad", "koi_srad"}.issubset(engineered.columns):
        radius_ratio = engineered["koi_prad"] / (engineered["koi_srad"] * 109.1)
        engineered["planet_star_radius_ratio"] = radius_ratio
        engineered["radius_ratio_squared"] = radius_ratio ** 2

    if {"koi_duration", "koi_period"}.issubset(engineered.columns):
        duration_ratio = engineered["koi_duration"] / (engineered["koi_period"] * 24)
        engineered["transit_duration_ratio"] = duration_ratio
        engineered["transit_duration_ratio_log"] = np.log1p(duration_ratio)

    if "koi_depth" in engineered.columns:
        engineered["depth_log"] = np.log1p(np.clip(engineered["koi_depth"], a_min=0, a_max=None))
        if "koi_prad" in engineered.columns:
            engineered["depth_radius_consistency"] = engineered["koi_depth"] / (
                engineered["koi_prad"] ** 2 + 1e-6
            )

    if {"koi_teq", "koi_steff"}.issubset(engineered.columns):
        engineered["temp_ratio"] = engineered["koi_teq"] / (engineered["koi_steff"] + 1e-6)
        engineered["temp_diff"] = engineered["koi_steff"] - engineered["koi_teq"]

    if "koi_period" in engineered.columns:
        engineered["period_log"] = np.log1p(np.clip(engineered["koi_period"], a_min=0, a_max=None))
        engineered["period_sqrt"] = np.sqrt(np.clip(engineered["koi_period"], a_min=0, a_max=None))
        engineered["period_category"] = pd.cut(
            engineered["koi_period"],
            bins=[0, 10, 50, 200, 1000],
            labels=[0, 1, 2, 3],
            include_lowest=True,
        )

    if {"koi_depth", "koi_duration"}.issubset(engineered.columns):
        signal_strength = engineered["koi_depth"] * engineered["koi_duration"]
        engineered["transit_signal_strength"] = signal_strength
        engineered["transit_signal_log"] = np.log1p(np.clip(signal_strength, a_min=0, a_max=None))

    if "koi_model_snr" in engineered.columns:
        engineered["snr_log"] = np.log1p(np.clip(engineered["koi_model_snr"], a_min=0, a_max=None))
        engineered["high_snr"] = (engineered["koi_model_snr"] > 20).astype(int)

    if "koi_insol" in engineered.columns:
        engineered["habitable_zone"] = (
            (engineered["koi_insol"] >= 0.25) & (engineered["koi_insol"] <= 2.0)
        ).astype(int)
        engineered["insol_log"] = np.log1p(np.clip(engineered["koi_insol"], a_min=0, a_max=None))

    if {"koi_srad", "koi_smass"}.issubset(engineered.columns):
        engineered["stellar_density"] = engineered["koi_smass"] / (
            engineered["koi_srad"] ** 3 + 1e-6
        )

    if "koi_impact" in engineered.columns:
        engineered["central_transit"] = (engineered["koi_impact"] < 0.5).astype(int)

    period_category = engineered.get("period_category")
    if isinstance(period_category, pd.Categorical):
        engineered["period_category"] = period_category.cat.codes.replace(-1, np.nan)

    for column in engineered.select_dtypes(include=["bool"]).columns:
        engineered[column] = engineered[column].astype(int)

    return engineered


def _prepare_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    engineered = _engineer_features(df)
    engineered = engineered.replace([np.inf, -np.inf], np.nan)

    feature_columns = [col for col in engineered.columns if col != "koi_disposition"]
    for column in feature_columns:
        if engineered[column].dtype.kind in {"O"}:  # ensure all features numeric
            engineered[column] = pd.to_numeric(engineered[column], errors="coerce")

    return engineered[feature_columns], (engineered["koi_disposition"] == "CONFIRMED").astype(int)


def _build_models(random_state: int = 42) -> Dict[str, object]:
    return {
        "LightGBM": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=12,
            num_leaves=50,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=12,
            min_child_weight=3,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=500,
            learning_rate=0.03,
            depth=10,
            l2_leaf_reg=3,
            random_seed=random_state,
            verbose=0,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=600,
            max_depth=35,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=600,
            max_depth=35,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=random_state,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=400,
            learning_rate=0.05,
            random_state=random_state,
        ),
    }


def train_ultimate_model(
    df: pd.DataFrame,
    *,
    test_size: float = 0.15,
    random_state: int = 42,
) -> TrainingResults:
    """Train the ExoStack ensemble on the supplied dataframe."""

    data = _remove_identifiers(df)
    available_features = [feature for feature in CRITICAL_FEATURES if feature in data.columns]

    if not available_features:
        raise ValueError("None of the critical features are present in the dataframe.")

    data = _fill_base_feature_gaps(data, available_features)
    data = _remove_outliers(data, available_features)

    # Subset the dataframe to the critical features plus the disposition label.
    subset_columns = available_features + ["koi_disposition"]
    data_subset = data[subset_columns].copy()

    X, y = _prepare_feature_matrix(data_subset)

    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    selector = SelectKBest(mutual_info_classif, k=max(1, int(len(feature_columns) * 0.8)))
    X_train_selected = selector.fit_transform(X_train_imputed, y_train)
    X_test_selected = selector.transform(X_test_imputed)

    selected_feature_names = [feature_columns[idx] for idx in selector.get_support(indices=True)]

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    resampler = SMOTEENN(random_state=random_state)
    X_resampled, y_resampled = resampler.fit_resample(X_train_scaled, y_train)

    models = _build_models(random_state)

    base_metrics: Dict[str, ModelMetrics] = {}
    trained_models: Dict[str, object] = {}

    for name, model in models.items():
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
        cv_scores = []
        for train_idx, valid_idx in cv.split(X_resampled, y_resampled):
            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_resampled[train_idx], y_resampled[train_idx])
            score = model_clone.score(X_resampled[valid_idx], y_resampled[valid_idx])
            cv_scores.append(score)
        cv_scores_arr = np.array(cv_scores)

        model.fit(X_resampled, y_resampled)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")

        base_metrics[name] = ModelMetrics(
            cv_mean=float(cv_scores_arr.mean()),
            cv_std=float(cv_scores_arr.std()),
            accuracy=float(accuracy_score(y_test, y_pred)),
            precision=float(precision_score(y_test, y_pred, zero_division=0)),
            recall=float(recall_score(y_test, y_pred, zero_division=0)),
            f1=float(f1_score(y_test, y_pred, zero_division=0)),
            auc=float(auc),
        )
        trained_models[name] = model

    top_models = sorted(base_metrics.items(), key=lambda item: item[1].accuracy, reverse=True)[:4]

    estimators = [
        (name.lower().replace(" ", "_"), trained_models[name])
        for name, _ in top_models
    ]

    stacking_final_estimator = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        random_state=random_state,
        verbose=-1,
    )

    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=stacking_final_estimator,
        cv=10,
        n_jobs=-1,
    )

    stacking_model.fit(X_resampled, y_resampled)
    y_stack_pred = stacking_model.predict(X_test_scaled)
    y_stack_proba = stacking_model.predict_proba(X_test_scaled)[:, 1]

    stacking_metrics = ModelMetrics(
        cv_mean=float("nan"),
        cv_std=float("nan"),
        accuracy=float(accuracy_score(y_test, y_stack_pred)),
        precision=float(precision_score(y_test, y_stack_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_stack_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_stack_pred, zero_division=0)),
        auc=float(roc_auc_score(y_test, y_stack_proba)),
    )

    artifacts = ModelArtifacts(
        feature_columns=feature_columns,
        imputer=imputer,
        selector=selector,
        scaler=scaler,
        selected_feature_names=selected_feature_names,
        trained_models=trained_models,
        stacking_model=stacking_model,
    )

    class_distribution = {
        "train": {
            "non_confirmed": int((y_resampled == 0).sum()),
            "confirmed": int((y_resampled == 1).sum()),
        },
        "test": {
            "non_confirmed": int((y_test == 0).sum()),
            "confirmed": int((y_test == 1).sum()),
        },
    }

    return TrainingResults(
        artifacts=artifacts,
        base_model_metrics=base_metrics,
        stacking_metrics=stacking_metrics,
        class_distribution=class_distribution,
        feature_count=len(feature_columns),
        X_test=X_test_scaled,
        y_test=y_test.values,
        selected_features=selected_feature_names,
    )


def _build_candidate_dataframe(params: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([params])


def predict_exoplanet_ultimate(
    artifacts: ModelArtifacts,
    *,
    orbital_period: float,
    transit_duration: float,
    transit_depth: float,
    planet_radius: float,
    equilibrium_temp: float,
    insolation_flux: float,
    stellar_temp: float,
    stellar_radius: float,
    stellar_mass: float = 1.0,
    stellar_logg: float = 4.5,
    impact_param: float = 0.5,
    model_snr: float = 10.0,
    planet_number: float = 1.0,
) -> Dict[str, float]:
    """Predict whether a candidate is a confirmed exoplanet."""

    params = {
        "koi_period": orbital_period,
        "koi_duration": transit_duration,
        "koi_depth": transit_depth,
        "koi_prad": planet_radius,
        "koi_teq": equilibrium_temp,
        "koi_insol": insolation_flux,
        "koi_steff": stellar_temp,
        "koi_srad": stellar_radius,
        "koi_smass": stellar_mass,
        "koi_slogg": stellar_logg,
        "koi_impact": impact_param,
        "koi_model_snr": model_snr,
        "koi_tce_plnt_num": planet_number,
    }

    candidate_df = _build_candidate_dataframe(params)
    engineered_candidate = _engineer_features(candidate_df)
    engineered_candidate = engineered_candidate.replace([np.inf, -np.inf], np.nan)

    candidate_aligned = engineered_candidate.reindex(columns=artifacts.feature_columns, fill_value=np.nan)
    candidate_imputed = artifacts.imputer.transform(candidate_aligned)
    candidate_selected = artifacts.selector.transform(candidate_imputed)
    candidate_scaled = artifacts.scaler.transform(candidate_selected)

    probabilities = artifacts.stacking_model.predict_proba(candidate_scaled)[0]
    predicted_class = int(probabilities[1] >= probabilities[0])

    return {
        "classification": "CONFIRMED EXOPLANET" if predicted_class == 1 else "NOT CONFIRMED",
        "confirmed_probability": float(probabilities[1]),
        "non_confirmed_probability": float(probabilities[0]),
    }
