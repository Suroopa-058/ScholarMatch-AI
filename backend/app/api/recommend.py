from fastapi import APIRouter, HTTPException, Request, status
from app.schemas.recommendation import RecommendationResponse
from app.schemas.student import StudentProfile

router = APIRouter(tags=["recommendations"])


@router.post("/recommend", response_model=RecommendationResponse)
def recommend(profile: StudentProfile, request: Request):
    service = request.app.state.recommendation_service
    if not service.ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={"message": "ML artifacts are not loaded. Export and place the trained models before requesting recommendations."})
    try:
        return {"student_id": profile.student_id, "recommendations": service.recommend(profile)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"message": str(exc)}) from exc
    except Exception as exc:
        request.app.state.logger.exception("Recommendation inference failed")
        raise HTTPException(status_code=500, detail={"message": "Unable to generate recommendations."}) from exc
