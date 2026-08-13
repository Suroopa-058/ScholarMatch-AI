from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_path: str, cache_path: Path, scholarships: list[dict]):
        self.model_path, self.cache_path, self.scholarships = model_path, cache_path, scholarships
        self.model = None
        self.scholarship_embeddings = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.scholarship_embeddings is not None

    def load(self) -> None:
        self.model = SentenceTransformer(self.model_path)
        if self.cache_path.exists():
            embeddings = np.load(self.cache_path)
            if embeddings.shape[0] != len(self.scholarships):
                raise ValueError("Cached scholarship embedding count does not match metadata.")
            self.scholarship_embeddings = embeddings
        else:
            texts = [f"{s['name']}. {s['description']}" for s in self.scholarships]
            self.scholarship_embeddings = self.model.encode(texts, normalize_embeddings=True)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_path, self.scholarship_embeddings)

    def similarities(self, student_text: str) -> np.ndarray:
        embedding = self.model.encode([student_text], normalize_embeddings=True)[0]
        return np.asarray(self.scholarship_embeddings @ embedding, dtype=float)
