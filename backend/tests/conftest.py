import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from app.ml.ensemble import Ensemble
from app.services.recommendation_service import RecommendationService


class FakeModel:
    def predict_proba(self, rows):
        # Test-only deterministic double; production never uses this class.
        return np.column_stack([np.zeros(len(rows)), np.linspace(0.1, 0.9, len(rows))])


class FakeEmbeddings:
    ready = True
    def similarities(self, _):
        return np.linspace(0.01, 0.10, 10)


class FakeExplanations:
    def explain(self, _):
        return [{"feature": "GPA", "impact": 0.2, "direction": "positive", "description": "Your gpa positively influenced this recommendation."}]


@pytest.fixture()
def client():
    app = create_app(load_models=False)
    artifacts = app.state.recommendation_service.artifacts
    artifacts.encoder = type("Encoder", (), {"transform": lambda self, values: [0 for _ in values]})()
    artifacts.ensemble = Ensemble({name: FakeModel() for name in ("xgboost", "random_forest", "lightgbm", "catboost")})
    service = RecommendationService(artifacts, FakeEmbeddings(), app.state.recommendation_service.scholarships, 5)
    app.state.recommendation_service = service
    app.state.explanation_service = FakeExplanations()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def valid_profile():
    return {"student_id": 12345, "semester": 6, "gpa": 7.67, "extracurricular_point": 46, "total_credits": 17, "class": "Computer Science", "has_failed_course": False, "student_year": 4}
