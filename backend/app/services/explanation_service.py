import numpy as np
import pandas as pd
import shap
from app.ml.preprocessing import FEATURE_NAMES

DISPLAY_NAMES = {"gpa": "GPA", "extracurricular_point": "Extracurricular Points", "total_credits": "Total Credits", "has_failed_course": "Failed Course", "student_year": "Student Year", "semester": "Semester", "class_encoded": "Class", "academic_weight": "Academic Weight", "extracurricular_weight": "Extracurricular Weight", "major_match": "Major Match", "semantic_similarity": "Semantic Similarity"}


class ExplanationService:
    def __init__(self, artifacts):
        self.artifacts = artifacts

    def explain(self, row: np.ndarray | pd.Series) -> list[dict]:
        values_row = row.to_numpy(dtype=float) if isinstance(row, pd.Series) else row
        feature_frame = pd.DataFrame([values_row], columns=FEATURE_NAMES)
        contributions = np.zeros(len(FEATURE_NAMES), dtype=float)
        # Each TreeExplainer uses the actual fitted model and feature vector; contributions are averaged with ensemble weights.
        for name, model in self.artifacts.ensemble.models.items():
            values = shap.TreeExplainer(model).shap_values(feature_frame)
            values = values[1] if isinstance(values, list) else values
            values = np.asarray(values)
            if values.ndim == 3:  # newer SHAP classifiers: samples, features, classes
                values = values[:, :, 1]
            contributions += self.artifacts.ensemble.weights[name] * values[0]
        output = []
        for feature, impact in zip(FEATURE_NAMES, contributions, strict=True):
            impact = float(impact)
            direction = "positive" if impact >= 0 else "negative"
            output.append({"feature": DISPLAY_NAMES[feature], "impact": impact, "direction": direction, "description": f"Your {DISPLAY_NAMES[feature].lower()} {'positively' if impact >= 0 else 'negatively'} influenced this recommendation."})
        return sorted(output, key=lambda item: abs(item["impact"]), reverse=True)
