"""Run in the Colab training environment after fitting the final pipeline.

Assign the exact fitted variables from the notebook below.  Do not fit or alter
them in this script: the API must receive the same models and class encoder
used for reported evaluation metrics.
"""
from pathlib import Path
import json
import joblib

OUTPUT = Path("models")


def export_models(xgboost_model, random_forest_model, lightgbm_model, catboost_model, class_encoder, ensemble_weights=None, sbert_model=None):
    targets = {"xgboost": xgboost_model, "random_forest": random_forest_model, "lightgbm": lightgbm_model, "catboost": catboost_model}
    for name, model in targets.items():
        destination = OUTPUT / name
        destination.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, destination / "model.joblib")
    (OUTPUT / "encoder").mkdir(parents=True, exist_ok=True)
    joblib.dump(class_encoder, OUTPUT / "encoder" / "class_encoder.joblib")
    if ensemble_weights is not None:
        required = set(targets)
        if set(ensemble_weights) != required:
            raise ValueError(f"ensemble_weights must contain exactly {sorted(required)}")
        (OUTPUT / "ensemble_weights.json").write_text(json.dumps(ensemble_weights, indent=2), encoding="utf-8")
    if sbert_model is not None:
        # Pass the fitted/selected SentenceTransformer used to create training embeddings.
        sbert_model.save(str(OUTPUT / "sbert" / "model"))


if __name__ == "__main__":
    raise SystemExit("Import export_models() into the training notebook and pass the five fitted artifacts.")
