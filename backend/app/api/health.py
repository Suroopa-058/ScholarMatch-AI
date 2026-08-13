from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request):
    service = request.app.state.recommendation_service
    return {"status": "healthy" if service.ready else "degraded", "service": "ScholarMatch AI Backend", "models_loaded": service.artifacts.ready, "sbert_loaded": service.embeddings.ready, "scholarships_loaded": len(service.scholarships) == 10, "startup_error": getattr(request.app.state, "startup_error", None)}
