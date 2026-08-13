import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_loader import load_models  # noqa: E402
from preprocessing import (  # noqa: E402
    build_student_text, build_feature_vector, calculate_major_match, FEATURE_COLUMNS,
)

SAMPLE_STUDENT = {
    "student_id": 999999,
    "semester": 8 % 2 + 1,  # keep within notebook's {1,2} semester domain -> 1
    "gpa": 8.5,
    "extracurricular_point": 70,
    "total_credits": 18,
    "class": "CS2021",
    "has_failed_course": False,
    "student_year": 4,
}


def test_student_text_format():
    text = build_student_text(SAMPLE_STUDENT)
    assert text.startswith("Major/Class: CS2021")
    assert "GPA: 8.5" in text
    assert "Extracurricular points: 70" in text
    assert "Total credits: 18" in text
    assert "Student year: 4" in text


def test_major_match_all():
    assert calculate_major_match("CS2021", "ALL") == 1.0


def test_major_match_specific():
    assert calculate_major_match("CS2021", "CS,IS,SE,DS,IT") == 1.0
    assert calculate_major_match("NS2013", "CS,IS,SE,DS,IT") == 0.0


def test_feature_vector_has_exactly_eleven_features_in_order():
    artifacts = load_models()
    scholarship = artifacts.scholarship_metadata[0]
    X, major_match = build_feature_vector(
        SAMPLE_STUDENT, scholarship, semantic_similarity=0.42, class_encoder=artifacts.class_encoder,
    )
    assert X.shape == (1, 11)
    assert len(FEATURE_COLUMNS) == 11
    assert FEATURE_COLUMNS == [
        "gpa", "extracurricular_point", "total_credits", "has_failed_course",
        "student_year", "semester", "class_encoded", "academic_weight",
        "extracurricular_weight", "major_match", "semantic_similarity",
    ]
