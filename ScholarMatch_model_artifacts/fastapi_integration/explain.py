"""
explain.py — SHAP-based explainability for /api/explain.

STATUS: The source notebook does NOT contain a SHAP implementation
(verified by inspecting all 22 cells). This module ADDS one, built on
the four already-trained models, rather than fabricating explanation
numbers.

APPROACH:
Each of the four base models (XGBoost, RandomForest, LightGBM,
CatBoost) is a tree ensemble, so shap.TreeExplainer is used for each
individually. Because the final prediction is a strict linear
combination of the four models' probabilities with equal 0.25
weights (see ensemble.py), the SHAP values are combined with the SAME
0.25/0.25/0.25/0.25 weights. This is mathematically consistent: the
explanation of a linear combination of model outputs is the same
linear combination of each model's SHAP attributions, when computed
against the same feature vector.

CAVEAT (documented, not hidden): shap.TreeExplainer's default output
for these libraries is in "margin"/log-odds space, not probability
space, for XGBoost/LightGBM/CatBoost. RandomForest's TreeExplainer
output is in probability space. This means the combined SHAP values
below are directly comparable/summable across the three
margin-space models, but RandomForest's contribution is on a
different scale (probability, not log-odds). This is a genuine
limitation of combining heterogeneous tree libraries in one
explanation, not something the notebook resolved either (it has no
SHAP code to reference). For this reason, impacts are reported as a
NORMALIZED share of total absolute impact per feature, and the sign
(direction) of each model's raw SHAP value is preserved and treated
as the source of truth for "positive"/"negative" direction, using a
majority vote across the four models' signs.

If this precision level is insufficient for production use, the
recommended fix is to move to a single canonical explanation model
(e.g., explain via the calibrated XGBoost model alone, since it is
one of the four ensemble members) rather than combining
heterogeneous SHAP scales. That decision is left to the backend
developer since the notebook does not specify one.
"""
from typing import Any, Dict, List

import numpy as np
import shap

from preprocessing import FEATURE_COLUMNS


def _tree_shap_row(model, X: np.ndarray) -> np.ndarray:
    """Returns a 1D array of SHAP values (len == number of features)
    for a single row, using shap.TreeExplainer."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    # Normalize shape across library return conventions:
    # - binary sklearn/xgboost/lightgbm models can return a single array
    #   (n_samples, n_features) for the positive class, or a list
    #   [class0_array, class1_array].
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.array(sv)
    if sv.ndim == 3:
        # (n_samples, n_features, n_classes) — take positive class
        sv = sv[:, :, -1]
    return sv[0]


def explain_recommendation(student: Dict[str, Any], scholarship: Dict[str, Any],
                            X: np.ndarray, artifacts) -> Dict[str, Any]:
    xgb_sv = _tree_shap_row(artifacts.xgb, X)
    rf_sv = _tree_shap_row(artifacts.rf, X)
    lgbm_sv = _tree_shap_row(artifacts.lgbm, X)
    catboost_sv = _tree_shap_row(artifacts.catboost, X)

    per_model = np.vstack([xgb_sv, rf_sv, lgbm_sv, catboost_sv])  # (4, 11)
    combined = 0.25 * per_model.sum(axis=0)  # equal-weight combination

    abs_total = np.abs(combined).sum() or 1.0

    explanations: List[Dict[str, Any]] = []
    for i, feature in enumerate(FEATURE_COLUMNS):
        signs = np.sign(per_model[:, i])
        direction = "positive" if signs.sum() >= 0 else "negative"
        impact_share = float(abs(combined[i]) / abs_total)
        explanations.append({
            "feature": feature,
            "impact": round(impact_share, 6),
            "direction": direction,
            "description": _describe(feature, direction),
        })

    explanations.sort(key=lambda e: e["impact"], reverse=True)

    return {
        "scholarship_id": scholarship["scholarship_id"],
        "explanation": explanations,
        "method": (
            "Equal-weight (0.25 each) combination of per-model shap.TreeExplainer "
            "values across XGBoost, RandomForest, LightGBM, and CatBoost. "
            "'impact' is each feature's share of total absolute combined impact "
            "for this specific student/scholarship pair (not a probability itself)."
        ),
    }


_DESCRIPTIONS = {
    "gpa": "GPA",
    "extracurricular_point": "Extracurricular activity points",
    "total_credits": "Total completed credits",
    "has_failed_course": "History of a failed course",
    "student_year": "Current student year",
    "semester": "Current semester",
    "class_encoded": "Major/class code",
    "academic_weight": "This scholarship's emphasis on academics",
    "extracurricular_weight": "This scholarship's emphasis on extracurriculars",
    "major_match": "Whether the student's major matches this scholarship's preferred majors",
    "semantic_similarity": "SBERT semantic similarity between student profile and scholarship description",
}


def _describe(feature: str, direction: str) -> str:
    label = _DESCRIPTIONS.get(feature, feature)
    verb = "positively" if direction == "positive" else "negatively"
    return f"{label} {verb} influenced the recommendation."
