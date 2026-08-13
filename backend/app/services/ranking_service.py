import numpy as np


def academic_fit(gpa: float, academic_weight: float) -> float:
    return float(np.clip((gpa / 10.0) * academic_weight, 0, 1))


def rank(scores: np.ndarray, context: list[dict], gpa: float, limit: int) -> list[dict]:
    ordered = sorted(enumerate(scores), key=lambda pair: float(pair[1]), reverse=True)[:limit]
    results = []
    for rank_number, (index, score) in enumerate(ordered, 1):
        item = context[index]
        scholarship = item["scholarship"]
        results.append({"rank": rank_number, "scholarship_id": scholarship["scholarship_id"], "name": scholarship["name"], "description": scholarship["description"], "recommendation_score": float(score), "semantic_similarity": item["semantic_similarity"], "major_match": item["major_match"], "eligible": item["eligible"], "academic_fit": academic_fit(gpa, scholarship["academic_weight"])})
    return results
