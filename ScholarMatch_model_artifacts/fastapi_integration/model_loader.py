"""
Loads all trained artifacts ONCE (e.g. at FastAPI startup via a
lifespan/startup event) and returns them for reuse across requests.

Nothing in here retrains, refits, or regenerates data. It only loads
what was produced by train_pipeline.py / the original notebook.
"""
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import joblib
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class LoadedArtifacts:
    xgb: Any
    rf: Any
    lgbm: Any
    catboost: Any
    class_encoder: Any
    scholarship_metadata: List[Dict[str, Any]]
    model_config: Dict[str, Any]
    sbert: Optional[Any]  # None if sentence-transformers/torch or network unavailable


def _resolve(*parts) -> str:
    """Resolve a path relative to this package, falling back to the
    top-level model_artifacts/ layout if files were copied as-is."""
    candidate = os.path.join(BASE_DIR, *parts)
    if os.path.exists(candidate):
        return candidate
    # fallback: ../models/... layout (model_artifacts/ top-level structure)
    alt = os.path.join(BASE_DIR, "..", *parts)
    return alt


def load_sbert(model_name: str):
    """
    Loads the SBERT model.

    Preference order:
      1. A local bundled copy at fastapi_integration/local_models/<model_name>/,
         if present. This package ships a verified local copy of
         'all-MiniLM-L6-v2' (see README: its embeddings were checked
         byte-for-byte against the notebook's own precomputed
         semantic_similarity values), so no network access is required
         at all for the default model.
      2. Otherwise, load by name from Hugging Face Hub (requires
         outbound network access on first run, or a local HF cache).

    Returns None (rather than raising) if neither works, so the rest of
    the service can still start; endpoints needing semantic_similarity
    should then fail with a clear error instead of fabricating a value.
    """
    try:
        from sentence_transformers import SentenceTransformer

        local_path = os.path.join(BASE_DIR, "local_models", model_name)
        if os.path.isdir(local_path):
            return SentenceTransformer(local_path)
        return SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001
        print(f"[model_loader] WARNING: could not load SBERT model '{model_name}': {exc}")
        return None


def load_models() -> LoadedArtifacts:
    models_dir = _resolve("models")

    xgb = XGBClassifier()
    xgb.load_model(os.path.join(models_dir, "xgb_model", "xgb_model.json"))

    rf = joblib.load(os.path.join(models_dir, "rf_model", "rf_model.joblib"))

    lgbm = joblib.load(os.path.join(models_dir, "lgbm_model", "lgbm_model.joblib"))

    catboost = CatBoostClassifier()
    catboost.load_model(os.path.join(models_dir, "catboost_model", "catboost_model.cbm"))

    class_encoder = joblib.load(_resolve("preprocessing", "class_encoder.joblib"))

    scholarships_path = _resolve("config", "scholarships.json")
    if not os.path.exists(scholarships_path):
        scholarships_path = _resolve("scholarships.json")
    with open(scholarships_path) as f:
        scholarship_metadata = json.load(f)

    config_path = _resolve("config", "model_config.json")
    if not os.path.exists(config_path):
        config_path = _resolve("model_config.json")
    with open(config_path) as f:
        model_config = json.load(f)

    sbert = load_sbert(model_config["sbert_model"])

    return LoadedArtifacts(
        xgb=xgb,
        rf=rf,
        lgbm=lgbm,
        catboost=catboost,
        class_encoder=class_encoder,
        scholarship_metadata=scholarship_metadata,
        model_config=model_config,
        sbert=sbert,
    )
