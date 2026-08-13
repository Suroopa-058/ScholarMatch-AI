"""
Pydantic request/response schemas for the ScholarMatch FastAPI backend.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class StudentProfile(BaseModel):
    student_id: int
    semester: int = Field(..., ge=1, le=2, description="1 or 2, as used in the notebook")
    gpa: float = Field(..., ge=0, le=10)
    extracurricular_point: float = Field(..., ge=0, le=100)
    total_credits: int = Field(..., ge=0)
    class_: str = Field(..., alias="class", description="Class/major code, e.g. 'CS2021'")
    has_failed_course: bool
    student_year: int = Field(..., ge=1)

    class Config:
        populate_by_name = True


class ScholarshipRecommendation(BaseModel):
    rank: int
    scholarship_id: str
    name: str
    description: str
    recommendation_score: float
    semantic_similarity: float
    major_match: int
    eligible: bool
    academic_fit: float


class RecommendationResponse(BaseModel):
    student_id: int
    recommendations: List[ScholarshipRecommendation]


class ExplainRequest(BaseModel):
    student: StudentProfile
    scholarship_id: str


class FeatureExplanation(BaseModel):
    feature: str
    impact: float
    direction: str
    description: str


class ExplainResponse(BaseModel):
    scholarship_id: str
    recommendation_score: float
    explanation: List[FeatureExplanation]
    method: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    sbert_loaded: bool
    num_scholarships: int
