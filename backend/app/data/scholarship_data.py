import json
from pathlib import Path

METADATA_PATH = Path(__file__).with_name("scholarship_metadata.json")


def load_scholarships() -> list[dict]:
    scholarships = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if len(scholarships) != 10 or len({item["scholarship_id"] for item in scholarships}) != 10:
        raise ValueError("Scholarship metadata must contain exactly the ten SCH001-SCH010 records.")
    # The exported notebook metadata stores major codes as CSV strings;
    # normalize once for the backend's typed feature service.
    for scholarship in scholarships:
        majors = scholarship["preferred_majors"]
        scholarship["preferred_majors"] = majors.split(",") if isinstance(majors, str) else majors
        scholarship["eligibility"] = {
            "min_gpa": 7, "min_extracurricular_point": 0,
            "min_total_credits": 14, "allow_failed_course": False,
            "min_student_year": 1, "max_student_year": 4,
        }
    return scholarships
