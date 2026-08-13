from pydantic import BaseModel, Field


class Recommendation(BaseModel):
    rank: int
    scholarship_id: str
    name: str
    description: str
    recommendation_score: float = Field(ge=0, le=1)
    semantic_similarity: float
    major_match: int
    eligible: bool
    academic_fit: float = Field(ge=0, le=1)


class RecommendationResponse(BaseModel):
    student_id: int
    recommendations: list[Recommendation]
