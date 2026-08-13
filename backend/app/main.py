import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api import auth, explain, health, profile, recommend
from app.config import settings
from app.data.scholarship_data import load_scholarships
from app.ml.model_loader import ModelArtifacts
from app.services.embedding_service import EmbeddingService
from app.services.explanation_service import ExplanationService
from app.services.recommendation_service import RecommendationService
from app.services.auth_service import AuthService


def create_app(load_models: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.auth_service.initialize()
        if load_models:
            try:
                app.state.recommendation_service.artifacts.load()
                app.state.recommendation_service.embeddings.load()
            except Exception as exc:  # service is intentionally degraded, never replaced by synthetic predictions
                app.state.startup_error = str(exc)
                app.state.logger.exception("ML startup failed")
        yield

    app = FastAPI(title=settings.app_name, version="1.0.0", description="SBERT + four-model ensemble scholarship ranking API.", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    app.state.logger = logging.getLogger("scholarmatch")
    scholarships = load_scholarships()
    artifacts = ModelArtifacts(settings.models_dir)
    embeddings = EmbeddingService(settings.sbert_model_path, settings.models_dir / "sbert" / "scholarship_embeddings.npy", scholarships)
    app.state.recommendation_service = RecommendationService(artifacts, embeddings, scholarships, settings.top_k)
    app.state.explanation_service = ExplanationService(artifacts)
    app.state.startup_error = None
    app.state.auth_service = AuthService(settings.database_path, settings.auth_secret, settings.auth_token_hours)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        errors = {".".join(str(part) for part in error["loc"] if part != "body"): error["msg"] for error in exc.errors()}
        return JSONResponse(status_code=422, content={"message": "Invalid student profile.", "errors": errors})

    app.include_router(health.router, prefix=settings.api_prefix)
    app.include_router(recommend.router, prefix=settings.api_prefix)
    app.include_router(explain.router, prefix=settings.api_prefix)
    app.include_router(auth.router, prefix=settings.api_prefix)
    app.include_router(profile.router, prefix=settings.api_prefix)

    @app.get(f"{settings.api_prefix}/scholarships", tags=["scholarships"])
    def scholarships_list():
        return scholarships

    @app.get(f"{settings.api_prefix}/scholarships/{{scholarship_id}}", tags=["scholarships"])
    def scholarship_detail(scholarship_id: str):
        for scholarship in scholarships:
            if scholarship["scholarship_id"] == scholarship_id:
                return scholarship
        return JSONResponse(status_code=404, content={"message": "Unknown scholarship ID."})

    return app


app = create_app()
