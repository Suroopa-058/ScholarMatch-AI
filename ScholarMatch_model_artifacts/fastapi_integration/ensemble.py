"""
Four-model equal-weight soft-voting ensemble.

Reproduces the notebook's CELL 16 EXACTLY:

    ensemble_probability = (
        0.25 * xgb_prob +
        0.25 * rf_prob +
        0.25 * lgbm_prob +
        0.25 * catboost_prob
    )

All four trained objects are binary classifiers exposing
`.predict_proba(X)` returning an (n_samples, 2) array, so column 1
(probability of class "1" / application_outcome == 1) is used for all
four models, matching the notebook's usage
(`model.predict_proba(X_test)[:, 1]`) precisely. This was verified by
inspecting the notebook's training cells for each model.
"""
import numpy as np

XGB_WEIGHT = 0.25
RF_WEIGHT = 0.25
LGBM_WEIGHT = 0.25
CATBOOST_WEIGHT = 0.25


def predict_ensemble_probability(models, X: np.ndarray) -> float:
    """
    models: object with .xgb, .rf, .lgbm, .catboost (as returned by
            model_loader.load_models())
    X: a single-row feature array shaped (1, 11), in FEATURE_COLUMNS order.
    Returns a single float probability in [0, 1].
    """
    xgb_prob = models.xgb.predict_proba(X)[:, 1][0]
    rf_prob = models.rf.predict_proba(X)[:, 1][0]
    lgbm_prob = models.lgbm.predict_proba(X)[:, 1][0]
    catboost_prob = models.catboost.predict_proba(X)[:, 1][0]

    ensemble_probability = (
        XGB_WEIGHT * xgb_prob
        + RF_WEIGHT * rf_prob
        + LGBM_WEIGHT * lgbm_prob
        + CATBOOST_WEIGHT * catboost_prob
    )
    return float(ensemble_probability)


def predict_all_probabilities(models, X: np.ndarray) -> dict:
    """Returns each model's individual probability plus the ensemble,
    useful for debugging/tests/explainability."""
    xgb_prob = float(models.xgb.predict_proba(X)[:, 1][0])
    rf_prob = float(models.rf.predict_proba(X)[:, 1][0])
    lgbm_prob = float(models.lgbm.predict_proba(X)[:, 1][0])
    catboost_prob = float(models.catboost.predict_proba(X)[:, 1][0])
    ensemble_probability = (
        XGB_WEIGHT * xgb_prob
        + RF_WEIGHT * rf_prob
        + LGBM_WEIGHT * lgbm_prob
        + CATBOOST_WEIGHT * catboost_prob
    )
    return {
        "xgboost": xgb_prob,
        "random_forest": rf_prob,
        "lightgbm": lgbm_prob,
        "catboost": catboost_prob,
        "ensemble": ensemble_probability,
    }
