"""
Eligibility logic — extracted verbatim from the training notebook,
CELL 8 ("CREATE MATCHING & ELIGIBILITY FEATURES"):

    interaction_df["eligible"] = (
        (interaction_df["gpa"] >= 7) &
        (interaction_df["total_credits"] >= 14) &
        (~interaction_df["has_failed_course"]) &
        (interaction_df["student_year"] <= 4)
    )

The rule is NOT scholarship-specific in the notebook — it is a single
global eligibility check applied identically to every scholarship.
This module reproduces that exact logic, unchanged, for a single
student (used at inference time instead of a full dataframe).
"""

MIN_GPA = 7
MIN_TOTAL_CREDITS = 14
MAX_STUDENT_YEAR = 4


def is_eligible(gpa: float, total_credits: int, has_failed_course: bool, student_year: int) -> bool:
    """Reproduces the notebook's eligibility rule for one student.

    Returns the same boolean the notebook would have computed for every
    row belonging to that student (eligibility does not vary by
    scholarship in the source notebook).
    """
    return bool(
        (gpa >= MIN_GPA)
        and (total_credits >= MIN_TOTAL_CREDITS)
        and (not has_failed_course)
        and (student_year <= MAX_STUDENT_YEAR)
    )
