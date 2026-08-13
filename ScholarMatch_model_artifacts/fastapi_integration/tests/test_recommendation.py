import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_loader import load_models  # noqa: E402
from preprocessing import build_feature_vector  # noqa: E402
from ensemble import predict_ensemble_probability  # noqa: E402
from recommendation_service import recommend_scholarships  # noqa: E402

SAMPLE_STUDENT = {
    "student_id": 999999,
    "semester": 1,
    "gpa": 8.5,
    "extracurricular_point": 70,
    "total_credits": 18,
    "class": "CS2021",
    "has_failed_course": False,
    "student_year": 4,
}


@pytest.fixture(scope="module")
def artifacts():
    return load_models()


def test_all_four_models_contribute_and_ensemble_is_averaged(artifacts):
    """Does not require SBERT: uses a placeholder semantic_similarity
    to validate that the ensemble math itself is correct and that all
    four models actually run."""
    scholarship = artifacts.scholarship_metadata[0]
    X, _ = build_feature_vector(SAMPLE_STUDENT, scholarship, semantic_similarity=0.45,
                                 class_encoder=artifacts.class_encoder)

    xgb_p = artifacts.xgb.predict_proba(X)[:, 1][0]
    rf_p = artifacts.rf.predict_proba(X)[:, 1][0]
    lgbm_p = artifacts.lgbm.predict_proba(X)[:, 1][0]
    cat_p = artifacts.catboost.predict_proba(X)[:, 1][0]

    expected = 0.25 * xgb_p + 0.25 * rf_p + 0.25 * lgbm_p + 0.25 * cat_p
    actual = predict_ensemble_probability(artifacts, X)

    assert math.isclose(expected, actual, rel_tol=1e-9)
    assert 0.0 <= actual <= 1.0


def test_full_pipeline_with_sbert(artifacts):
    if artifacts.sbert is None:
        pytest.skip("SBERT model not available in this environment (no network access to Hugging Face Hub).")

    result = recommend_scholarships(SAMPLE_STUDENT, artifacts, top_k=5)

    assert result["student_id"] == SAMPLE_STUDENT["student_id"]
    recs = result["recommendations"]

    # Exactly 5 recommendations
    assert len(recs) == 5

    # Sorted descending by recommendation_score
    scores = [r["recommendation_score"] for r in recs]
    assert scores == sorted(scores, reverse=True)

    # Ranks are 1..5
    assert [r["rank"] for r in recs] == [1, 2, 3, 4, 5]

    # No NaNs anywhere
    flat = json.dumps(recs)
    assert "NaN" not in flat

    # JSON serializable
    json.loads(flat)
