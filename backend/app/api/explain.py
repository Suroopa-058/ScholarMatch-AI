from fastapi import APIRouter, HTTPException, Request, status
from app.schemas.explanation import ExplanationRequest, ExplanationResponse

router = APIRouter(tags=["explanations"])


@router.post("/explain", response_model=ExplanationResponse)
def explain(payload: ExplanationRequest, request: Request):
    service = request.app.state.recommendation_service
    if not service.ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"message": "ML artifacts are not loaded."})
    if payload.scholarship_id not in {s["scholarship_id"] for s in service.scholarships}:
        raise HTTPException(status_code=404, detail={"message": "Unknown scholarship ID."})
    row = service.cached_features(payload.student_id, payload.scholarship_id)
    if row is None:
        raise HTTPException(status_code=404, detail={"message": "Student profile is not available. Call /api/recommend for this student before requesting an explanation."})
    try:
        return {"scholarship_id": payload.scholarship_id, "explanation": request.app.state.explanation_service.explain(row)}
    except Exception as exc:
        request.app.state.logger.exception("Explanation generation failed")
        raise HTTPException(status_code=500, detail={"message": "Unable to generate explanation."}) from exc
