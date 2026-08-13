"""
recommendation_service.py

Exposes recommend_scholarships(student_profile, artifacts) which runs
the full inference pipeline described in the notebook, for one NEW
student, against all 10 scholarships, and returns the Top 5 ranked by
recommendation_score (the ensemble probability).

Pipeline (mirrors the notebook end-to-end, inference-only):
  1. Validate student profile (via schemas.StudentProfile)
  2. Build student_text (CELL 5 formula)
  3. Generate SBERT embedding for student_text + cosine similarity vs.
     each of the 10 precomputed/recomputed scholarship_text embeddings
     (CELL 6 / CELL 7 formulas)
  4. major_match (CELL 8)
  5. eligible (CELL 8 / eligibility.py)
  6. Assemble the exact 11-feature vector (CELL 11 order)
  7. Run all four trained models' predict_proba
  8. 25/25/25/25 soft voting (CELL 16 / ensemble.py)
  9. Rank all 10 scholarships by recommendation_score, descending
  10. Return Top 5 with rank, plus supporting fields actually computed
      by the notebook (semantic_similarity, major_match, eligible,
      academic_fit == the notebook's "academic_component").

NOTE on academic_fit: the notebook does not compute a field literally
named "academic_fit". It computes `academic_component = gpa_score *
academic_weight` (CELL 9), which is the actual, already-defined
per-scholarship academic-fit quantity in the source notebook. That is
what is returned here as "academic_fit" -- nothing is invented.
"""
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from eligibility import is_eligible
from preprocessing import build_student_text, build_scholarship_text, build_feature_vector
from ensemble import predict_ensemble_probability


def recommend_scholarships(student: Dict[str, Any], artifacts, top_k: int = 5) -> Dict[str, Any]:
    if artifacts.sbert is None:
        raise RuntimeError(
            "SBERT model is not loaded. semantic_similarity cannot be computed for a "
            "new student without it. Ensure the backend has network access to download "
            "'all-MiniLM-L6-v2' from Hugging Face on first run (or provide a local cache)."
        )

    student_text = build_student_text(student)
    student_embedding = artifacts.sbert.encode([student_text])

    scholarships = artifacts.scholarship_metadata
    scholarship_texts = [build_scholarship_text(s) for s in scholarships]
    scholarship_embeddings = artifacts.sbert.encode(scholarship_texts)

    similarities = cosine_similarity(student_embedding, scholarship_embeddings)[0]

    eligible = is_eligible(
        gpa=student["gpa"],
        total_credits=student["total_credits"],
        has_failed_course=student["has_failed_course"],
        student_year=student["student_year"],
    )

    gpa_score = float(student["gpa"]) / 10.0

    results: List[Dict[str, Any]] = []
    for scholarship, sim in zip(scholarships, similarities):
        X, major_match = build_feature_vector(
            student=student,
            scholarship=scholarship,
            semantic_similarity=float(sim),
            class_encoder=artifacts.class_encoder,
        )
        score = predict_ensemble_probability(artifacts, X)
        academic_fit = gpa_score * float(scholarship["academic_weight"])  # notebook's academic_component

        results.append({
            "scholarship_id": scholarship["scholarship_id"],
            "name": scholarship["name"],
            "description": scholarship["description"],
            "recommendation_score": round(score, 6),
            "semantic_similarity": round(float(sim), 6),
            "major_match": int(major_match),
            "eligible": eligible,
            "academic_fit": round(academic_fit, 6),
        })

    results.sort(key=lambda r: r["recommendation_score"], reverse=True)

    top = results[:top_k]
    for i, r in enumerate(top, start=1):
        r["rank"] = i

    return {
        "student_id": student["student_id"],
        "recommendations": top,
    }
