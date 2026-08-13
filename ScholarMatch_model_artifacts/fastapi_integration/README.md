# ScholarMatch AI — FastAPI Integration Package

Self-contained: this folder has its own copies of the trained models,
class encoder, scholarship metadata, **and a verified local copy of the
SBERT model** (`local_models/all-MiniLM-L6-v2/`), so it can be dropped
straight into a FastAPI project and run with **no network access
required**.

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

That's it — SBERT loads from `local_models/` by default. (If you'd
rather always pull the latest weights from Hugging Face Hub instead,
delete the `local_models/` folder; `model_loader.py` will fall back to
downloading `all-MiniLM-L6-v2` by name.)

## Endpoints

- `GET /api/health` — reports whether models + SBERT loaded, and how
  many scholarships are configured.
- `POST /api/recommend` — body is a student profile (see
  `schemas.StudentProfile`), returns Top-5 ranked scholarships.
- `POST /api/explain` — body is `{"student": {...}, "scholarship_id":
  "SCH001"}`, returns SHAP-based feature explanations for that
  student/scholarship pair.

## Example request

```bash
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": 999999,
    "semester": 1,
    "gpa": 8.5,
    "extracurricular_point": 70,
    "total_credits": 18,
    "class": "CS2021",
    "has_failed_course": false,
    "student_year": 4
  }'
```

## Files

| File | Purpose |
|---|---|
| `model_loader.py` | loads all 4 models, class encoder, scholarship metadata, and SBERT — once |
| `preprocessing.py` | student_text/scholarship_text, major_match, class encoding, 11-feature vector assembly |
| `eligibility.py` | notebook's eligibility rule, unchanged |
| `ensemble.py` | 25/25/25/25 soft-voting, exactly as the notebook |
| `recommendation_service.py` | full pipeline: new student → Top-5 ranked scholarships |
| `explain.py` | SHAP explanations (added — the notebook has none); see its docstring for method + caveats |
| `schemas.py` | pydantic request/response models |
| `app.py` | reference FastAPI app wiring the above to the 3 endpoints |
| `tests/` | pytest suite — model loading, preprocessing, and end-to-end recommendation |

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

`test_full_pipeline_with_sbert` uses the bundled local model and should
pass out of the box, with no network access needed.

## Do not

- Do not have the API load `scholarship_interactions_92900_FULL.csv` —
  it's a training artifact only, not needed for inference.
- Do not refit `class_encoder` — a genuinely new/unseen class code
  should be surfaced as an error to the caller, not silently encoded.
- Do not retrain models inside request handlers.
