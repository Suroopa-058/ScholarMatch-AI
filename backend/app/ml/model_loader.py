import json
from pathlib import Path
import joblib
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from .ensemble import Ensemble


class ModelArtifacts:
    """Loads exactly the models/encoder exported by the training notebook once."""
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.encoder = None
        self.ensemble = None

    @property
    def ready(self) -> bool:
        return self.encoder is not None and self.ensemble is not None

    def load(self) -> None:
        files = {
            "xgboost": self.models_dir / "xgboost" / "model.json",
            "random_forest": self.models_dir / "random_forest" / "model.joblib",
            "lightgbm": self.models_dir / "lightgbm" / "model.joblib",
            "catboost": self.models_dir / "catboost" / "model.cbm",
            "encoder": self.models_dir / "encoder" / "class_encoder.joblib",
        }
        absent = [str(path.relative_to(self.models_dir)) for path in files.values() if not path.exists()]
        if absent:
            raise FileNotFoundError("Missing exported model artifacts: " + ", ".join(absent))
        self.encoder = joblib.load(files.pop("encoder"))
        xgboost = XGBClassifier()
        xgboost.load_model(files["xgboost"])
        catboost = CatBoostClassifier()
        catboost.load_model(files["catboost"])
        models = {
            "xgboost": xgboost,
            "random_forest": joblib.load(files["random_forest"]),
            "lightgbm": joblib.load(files["lightgbm"]),
            "catboost": catboost,
        }
        weights_path = self.models_dir / "ensemble_weights.json"
        weights = json.loads(weights_path.read_text()) if weights_path.exists() else None
        self.ensemble = Ensemble(models, weights)
