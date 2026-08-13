import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_loader import load_models  # noqa: E402


def test_all_four_models_load():
    artifacts = load_models()
    assert artifacts.xgb is not None
    assert artifacts.rf is not None
    assert artifacts.lgbm is not None
    assert artifacts.catboost is not None


def test_class_encoder_loads():
    artifacts = load_models()
    assert artifacts.class_encoder is not None
    assert len(artifacts.class_encoder.classes_) > 0


def test_scholarship_metadata_has_ten_scholarships():
    artifacts = load_models()
    assert len(artifacts.scholarship_metadata) == 10
    ids = {s["scholarship_id"] for s in artifacts.scholarship_metadata}
    assert ids == {f"SCH{str(i).zfill(3)}" for i in range(1, 11)}


def test_sbert_loads_or_warns_gracefully():
    artifacts = load_models()
    # sbert may be None in offline environments; loader must not crash either way
    assert artifacts.sbert is None or hasattr(artifacts.sbert, "encode")
