from app.schemas.student import StudentProfile
from .feature_service import build_features, profile_text
from .ranking_service import rank


class RecommendationService:
    def __init__(self, artifacts, embeddings, scholarships, top_k: int):
        self.artifacts, self.embeddings, self.scholarships, self.top_k = artifacts, embeddings, scholarships, top_k
        self.profile_cache: dict[int, StudentProfile] = {}
        self.feature_cache: dict[tuple[int, str], object] = {}

    @property
    def ready(self) -> bool:
        return self.artifacts.ready and self.embeddings.ready

    def candidate_features(self, profile: StudentProfile):
        similarities = self.embeddings.similarities(profile_text(profile))
        return build_features(profile, self.scholarships, similarities, self.artifacts.encoder)

    def recommend(self, profile: StudentProfile) -> list[dict]:
        features, context = self.candidate_features(profile)
        scores = self.artifacts.ensemble.predict_proba(features)
        self.profile_cache[profile.student_id] = profile
        self.feature_cache.update({(profile.student_id, item["scholarship"]["scholarship_id"]): features.iloc[index] for index, item in enumerate(context)})
        return rank(scores, context, profile.gpa, self.top_k)

    def cached_features(self, student_id: int, scholarship_id: str):
        return self.feature_cache.get((student_id, scholarship_id))
