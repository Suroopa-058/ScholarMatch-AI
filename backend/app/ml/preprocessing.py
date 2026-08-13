FEATURE_NAMES = ["gpa", "extracurricular_point", "total_credits", "has_failed_course", "student_year", "semester", "class_encoded", "academic_weight", "extracurricular_weight", "major_match", "semantic_similarity"]


def encode_class(encoder, student_class: str) -> float:
    """Apply the exported training encoder; unknown categories are an export/configuration error."""
    try:
        return float(encoder.transform([student_class])[0])
    except ValueError as exc:
        raise ValueError(f"Class '{student_class}' was not seen by the training encoder.") from exc
