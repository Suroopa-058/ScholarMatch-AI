# ScholarMatch AI backend

FastAPI backend for the ScholarMatch AI React frontend. It scores all ten fixed scholarships with the **trained** SBERT + XGBoost, Random Forest, LightGBM, and CatBoost pipeline, ranks them, and derives explanations from actual SHAP values.

> The supplied React repository currently contains only a landing page. It has no service files, profile form, recommendation UI, explanation UI, scholarship-detail UI, or API calls. This backend implements the request/response contract specified for those planned frontend services without changing the frontend.

## Architecture

```text
React Frontend
      ↓
FastAPI REST API
      ↓
Recommendation Service
      ↓
Feature Engineering
      ↓
SBERT Semantic Similarity
      ↓
XGBoost + Random Forest + LightGBM + CatBoost
      ↓
Ensemble Score
      ↓
Ranking Engine
      ↓
Top-K Scholarships
      ↓
SHAP Explanation
```

## Install and run

From `backend/` (Python 3.10+ recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload --port 8000
```

Swagger is at `http://localhost:8000/docs`; health is at `http://localhost:8000/api/health`.

Set the frontend root `.env` to:

```env
VITE_API_BASE_URL=http://localhost:8000/api
```

## Required trained artifacts

The API intentionally serves no synthetic scores. Until artifacts are exported, `/api/health` is `degraded` and inference returns HTTP 503. Run `export_models.py` from the training/Colab environment using the exact fitted objects, then copy the output tree into `backend/models/`:

```text
models/
  xgboost/model.joblib
  random_forest/model.joblib
  lightgbm/model.joblib
  catboost/model.joblib
  encoder/class_encoder.joblib
  ensemble_weights.json                 # optional; equal weights if omitted
  sbert/model/                          # optional local copy of exact training SBERT model
```

If `sbert/model/` is exported, set `SBERT_MODEL_PATH=models/sbert/model` (use an absolute path if starting outside `backend`). Otherwise set the exact Hugging Face identifier/revision used during training. The service creates and reuses `models/sbert/scholarship_embeddings.npy`; delete it after changing the SBERT model or scholarship descriptions.

`class_encoder.joblib` must be the exact encoder trained on the `class` field. The model feature order is fixed: `gpa`, `extracurricular_point`, `total_credits`, `has_failed_course`, `student_year`, `semester`, `class_encoded`, `academic_weight`, `extracurricular_weight`, `major_match`, `semantic_similarity`.

The ten records in `app/data/scholarship_metadata.json` are the only candidates. Their descriptions, preferences, weights, and eligibility thresholds must be reconciled with the training notebook before final evaluation, because no original metadata/training notebook was supplied in this repository.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Startup/artifact readiness |
| `POST /api/recommend` | Scores all 10 candidates and returns ranked top 5 |
| `POST /api/explain` | SHAP contributions for a scholarship from the most recent recommendation request for that student ID |
| `GET /api/scholarships` | Fixed scholarship metadata |
| `GET /api/scholarships/{id}` | One scholarship record |

`POST /api/recommend` accepts the documented student profile (including JSON key `class`) and returns each result’s `rank`, score, similarity, major match, eligibility, and academic fit. Eligibility is returned but does not overwrite or fabricate model scores. `POST /api/explain` accepts `{ "student_id": 12345, "scholarship_id": "SCH001" }`; request recommendations first so the exact feature vector is available.

SHAP uses a `TreeExplainer` over each actual fitted tree model, then combines per-feature contributions with the same ensemble weights used for prediction. No explanation values are stored or hardcoded.

## Tests

```powershell
pytest
```

Tests use isolated test doubles only to exercise HTTP contracts. Production startup loads only exported artifacts and never falls back to those doubles.
