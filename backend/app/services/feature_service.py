import numpy as np
import pandas as pd
from app.ml.preprocessing import FEATURE_NAMES, encode_class
from app.schemas.student import StudentProfile


def profile_text(profile: StudentProfile) -> str:
    return (f"Major/Class: {profile.class_}. GPA: {round(profile.gpa, 2)}. "
            f"Extracurricular points: {profile.extracurricular_point}. "
            f"Total credits: {profile.total_credits}. Student year: {profile.student_year}.")


def major_match(student_class: str, majors: list[str]) -> int:
    if "ALL" in {major.upper() for major in majors}:
        return 1
    normalized = student_class.casefold()
    return int(any(major.casefold() in normalized or normalized in major.casefold() for major in majors))


def eligible(profile: StudentProfile, requirements: dict) -> bool:
    return (profile.gpa >= requirements["min_gpa"] and
            profile.extracurricular_point >= requirements["min_extracurricular_point"] and
            profile.total_credits >= requirements["min_total_credits"] and
            profile.student_year >= requirements["min_student_year"] and
            profile.student_year <= requirements.get("max_student_year", float("inf")) and
            (requirements["allow_failed_course"] or not profile.has_failed_course))


def build_features(profile: StudentProfile, scholarships: list[dict], similarities: np.ndarray, encoder) -> tuple[pd.DataFrame, list[dict]]:
    class_encoded = encode_class(encoder, profile.class_)
    rows, context = [], []
    for scholarship, similarity in zip(scholarships, similarities, strict=True):
        match = major_match(profile.class_, scholarship["preferred_majors"])
        row = [profile.gpa, profile.extracurricular_point, profile.total_credits, int(profile.has_failed_course), profile.student_year, profile.semester, class_encoded, scholarship["academic_weight"], scholarship["extracurricular_weight"], match, float(similarity)]
        rows.append(row)
        context.append({"scholarship": scholarship, "semantic_similarity": float(similarity), "major_match": match, "eligible": eligible(profile, scholarship["eligibility"])})
    # The exported models were trained with named DataFrame columns.
    return pd.DataFrame(rows, columns=FEATURE_NAMES, dtype=float), context


def feature_mapping(row: np.ndarray) -> dict[str, float]:
    return dict(zip(FEATURE_NAMES, row.tolist(), strict=True))
