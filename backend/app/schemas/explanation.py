from pydantic import BaseModel, Field


class ExplanationRequest(BaseModel):
    student_id: int = Field(gt=0)
    scholarship_id: str = Field(min_length=1, max_length=50)


class FeatureExplanation(BaseModel):
    feature: str
    impact: float
    direction: str
    description: str


class ExplanationResponse(BaseModel):
    scholarship_id: str
    explanation: list[FeatureExplanation]
