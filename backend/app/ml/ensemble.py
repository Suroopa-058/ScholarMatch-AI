import numpy as np


class Ensemble:
    def __init__(self, models: dict, weights: dict[str, float] | None = None):
        self.models = models
        weights = weights or {name: 1.0 for name in models}
        missing = set(models) - set(weights)
        if missing:
            raise ValueError(f"Missing ensemble weights for: {sorted(missing)}")
        total = sum(weights[name] for name in models)
        if total <= 0:
            raise ValueError("Ensemble weights must sum to a positive value.")
        self.weights = {name: weights[name] / total for name in models}

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = []
        for name, model in self.models.items():
            if not hasattr(model, "predict_proba"):
                raise TypeError(f"{name} does not expose predict_proba.")
            probabilities.append(self.weights[name] * np.asarray(model.predict_proba(features))[:, 1])
        return np.sum(probabilities, axis=0)
