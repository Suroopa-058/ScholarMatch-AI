from fastapi import APIRouter, Header, HTTPException, Request, Response, status
from app.schemas.student import StudentProfile
from app.services.auth_service import AuthError

router = APIRouter(prefix="/profile", tags=["saved profile"])


def user_id(request: Request, authorization: str | None) -> int:
    try:
        return request.app.state.auth_service.user_from_token(authorization)["id"]
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail={"message": str(exc)}) from exc


@router.get("/me", response_model=StudentProfile | None)
def get_profile(request: Request, authorization: str | None = Header(default=None)):
    return request.app.state.auth_service.load_profile(user_id(request, authorization))


@router.put("/me", status_code=status.HTTP_204_NO_CONTENT)
def save_profile(profile: StudentProfile, request: Request, authorization: str | None = Header(default=None)):
    request.app.state.auth_service.save_profile(user_id(request, authorization), profile.model_dump(by_alias=True))
    return Response(status_code=status.HTTP_204_NO_CONTENT)
