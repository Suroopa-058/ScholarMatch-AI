"""
Minimal FastAPI app wiring the artifacts to the three required
endpoints. This is a reference/starter implementation — adapt
error handling, logging, and auth to your production standards.

Run:
    uvicorn app:app --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from model_loader import load_models
from schemas import (
    StudentProfile, RecommendationResponse, ExplainRequest,
    ExplainResponse, HealthResponse,
)
from recommendation_service import recommend_scholarships
from preprocessing import build_feature_vector
from ensemble import predict_ensemble_probability
from explain import explain_recommendation

ARTIFACTS = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load everything ONCE at startup, not per-request.
    ARTIFACTS["models"] = load_models()
    yield
    ARTIFACTS.clear()


app = FastAPI(title="ScholarMatch AI Backend", lifespan=lifespan)


@app.get("/api/health", response_model=HealthResponse)
def health():
    artifacts = ARTIFACTS.get("models")
    return HealthResponse(
        status="ok" if artifacts else "not_ready",
        models_loaded=artifacts is not None,
        sbert_loaded=bool(artifacts and artifacts.sbert is not None),
        num_scholarships=len(artifacts.scholarship_metadata) if artifacts else 0,
    )


@app.post("/api/recommend", response_model=RecommendationResponse)
def recommend(student: StudentProfile):
    artifacts = ARTIFACTS["models"]
    try:
        result = recommend_scholarships(student.model_dump(by_alias=True), artifacts, top_k=5)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.post("/api/explain", response_model=ExplainResponse)
def explain(payload: ExplainRequest):
    artifacts = ARTIFACTS["models"]
    scholarship = next(
        (s for s in artifacts.scholarship_metadata if s["scholarship_id"] == payload.scholarship_id),
        None,
    )
    if scholarship is None:
        raise HTTPException(status_code=404, detail=f"Unknown scholarship_id '{payload.scholarship_id}'")

    if artifacts.sbert is None:
        raise HTTPException(status_code=503, detail="SBERT model not loaded; cannot compute semantic_similarity.")

    student = payload.student.model_dump(by_alias=True)
    from preprocessing import build_student_text, build_scholarship_text
    from sklearn.metrics.pairwise import cosine_similarity

    student_text = build_student_text(student)
    scholarship_text = build_scholarship_text(scholarship)
    student_emb = artifacts.sbert.encode([student_text])
    scholarship_emb = artifacts.sbert.encode([scholarship_text])
    similarity = float(cosine_similarity(student_emb, scholarship_emb)[0][0])

    X, _ = build_feature_vector(student, scholarship, similarity, artifacts.class_encoder)
    score = predict_ensemble_probability(artifacts, X)

    result = explain_recommendation(student, scholarship, X, artifacts)
    result["recommendation_score"] = round(score, 6)
    return result
