"""
Preprocessing for NEW students at inference time.

Reproduces, unchanged, the logic from:
  - CELL 5  (student_text / scholarship_text construction)
  - CELL 8  (major_match calculation)
  - CELL 11 (class_encoded via the fitted LabelEncoder, final feature order)

No re-fitting happens here. The LabelEncoder used below is the exact
object fitted in the training notebook and must be loaded from
preprocessing/class_encoder.joblib (see model_loader.py).
"""
from typing import Dict, Any
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "gpa",
    "extracurricular_point",
    "total_credits",
    "has_failed_course",
    "student_year",
    "semester",
    "class_encoded",
    "academic_weight",
    "extracurricular_weight",
    "major_match",
    "semantic_similarity",
]


def build_student_text(student: Dict[str, Any]) -> str:
    """Exact reproduction of CELL 5's student_text formula."""
    return (
        f"Major/Class: {student['class']}"
        f". GPA: {round(float(student['gpa']), 2)}"
        f". Extracurricular points: {student['extracurricular_point']}"
        f". Total credits: {student['total_credits']}"
        f". Student year: {student['student_year']}"
    )


def build_scholarship_text(scholarship: Dict[str, Any]) -> str:
    """Exact reproduction of CELL 5's scholarship_text formula."""
    return f"{scholarship['name']}. {scholarship['description']}"


def calculate_major_match(student_class: str, preferred_majors: str) -> float:
    """Exact reproduction of CELL 8's calculate_major_match()."""
    if preferred_majors == "ALL":
        return 1.0
    preferred_list = preferred_majors.split(",")
    return 1.0 if student_class[:2] in preferred_list else 0.0


def encode_class(class_encoder, student_class: str) -> int:
    """
    Transform a class/major code using the FITTED encoder from training.

    If a brand-new class code appears that the encoder never saw during
    training, scikit-learn's LabelEncoder will raise a ValueError. We
    surface that clearly rather than silently refitting (refitting here
    would violate "no data leakage" / "same encoding" requirements).
    """
    student_class = str(student_class)
    try:
        return int(class_encoder.transform([student_class])[0])
    except ValueError as exc:
        known = ", ".join(sorted(map(str, class_encoder.classes_))[:10])
        raise ValueError(
            f"Unknown class/major code '{student_class}'. This encoder was fit "
            f"only on class codes seen during training (examples: {known}, ...). "
            "A new class code cannot be encoded without retraining."
        ) from exc


def build_feature_vector(
    student: Dict[str, Any],
    scholarship: Dict[str, Any],
    semantic_similarity: float,
    class_encoder,
) -> np.ndarray:
    """
    Assemble the exact 11-feature vector, in the exact order required
    by the trained models (see config/feature_columns.json).
    """
    major_match = calculate_major_match(student["class"], scholarship["preferred_majors"])
    class_encoded = encode_class(class_encoder, student["class"])

    row = {
        "gpa": float(student["gpa"]),
        "extracurricular_point": float(student["extracurricular_point"]),
        "total_credits": float(student["total_credits"]),
        "has_failed_course": int(bool(student["has_failed_course"])),
        "student_year": float(student["student_year"]),
        "semester": float(student["semester"]),
        "class_encoded": float(class_encoded),
        "academic_weight": float(scholarship["academic_weight"]),
        "extracurricular_weight": float(scholarship["extracurricular_weight"]),
        "major_match": float(major_match),
        "semantic_similarity": float(semantic_similarity),
    }

    # Returned as a DataFrame (not a bare ndarray) because all four models
    # were trained on pandas DataFrames with these exact column names
    # (X_train[FEATURES]); scikit-learn/XGBoost/LightGBM/CatBoost all
    # warn or can behave inconsistently if given unnamed arrays instead.
    X = pd.DataFrame([[row[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    return X, major_match
