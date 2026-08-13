from pydantic import BaseModel, ConfigDict, Field, field_validator


class StudentProfile(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    student_id: int = Field(gt=0, examples=[12345])
    semester: int = Field(ge=1, le=12, examples=[6])
    gpa: float = Field(ge=0, le=10, examples=[7.67])
    extracurricular_point: float = Field(ge=0, examples=[46])
    total_credits: float = Field(gt=0, examples=[17])
    class_: str = Field(alias="class", min_length=1, max_length=100, examples=["Computer Science"])
    has_failed_course: bool
    student_year: int = Field(ge=1, le=8, examples=[4])

    @field_validator("class_")
    @classmethod
    def non_blank_class(cls, value: str) -> str:
        if not value:
            raise ValueError("Class must not be blank.")
        return value
