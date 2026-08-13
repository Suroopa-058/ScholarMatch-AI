from fastapi import APIRouter, HTTPException, Request, status
from app.schemas.auth import AuthResponse, LoginRequest, SignUpRequest
from app.services.auth_service import AuthError

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def signup(payload: SignUpRequest, request: Request):
    try:
        return request.app.state.auth_service.signup(payload.name, str(payload.email), payload.password)
    except AuthError as exc:
        raise HTTPException(status_code=409, detail={"message": str(exc)}) from exc


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, request: Request):
    try:
        return request.app.state.auth_service.login(str(payload.email), payload.password)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail={"message": str(exc)}) from exc
