# ScholarMatch AI

AI-powered scholarship recommendation system for a final-year project. The repository contains a React + TypeScript + Vite landing frontend and a production-oriented FastAPI inference backend.

The backend is in [backend/README.md](backend/README.md). It provides SBERT semantic matching, the exported four-model ensemble (XGBoost, Random Forest, LightGBM, CatBoost), rank-based top-five recommendations, eligibility reporting, and real SHAP feature explanations.

## Run

Start the backend from `backend/` after installing its requirements:

```powershell
uvicorn app.main:app --reload --port 8000
```

Set the root frontend environment variable:

```env
VITE_API_BASE_URL=http://localhost:8000/api
```

Then run the existing frontend normally:

```powershell
npm install
npm run dev
```

The actual trained artifacts must be exported into `backend/models/` before inference can run. Until then, the backend reports a degraded but healthy HTTP service and safely rejects model requests rather than returning invented recommendations.
"# ScholarMatch-AI" 
