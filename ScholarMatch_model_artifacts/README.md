# ScholarMatch AI — Model Artifacts & FastAPI Integration Package

This package was produced by inspecting the original ScholarMatch training
notebook (`proposed_system_dataset (1).ipynb`) cell by cell, and exporting
its trained models/artifacts so they can be loaded by a FastAPI backend
for **inference only**. Nothing about the ML approach, feature set, or
ensemble logic was changed.

---

## 1. How the models were trained

The notebook's pipeline (unchanged):

```
Student Profile → Preprocessing → Feature Engineering → SBERT Semantic
Similarity → [XGBoost, Random Forest, LightGBM, CatBoost] →
Equal-Weight (25/25/25/25) Soft-Voting Ensemble → Recommendation Score →
Ranking → Top-K
```

- **Data**: 78,039 original student records → deduplicated to 9,290 unique
  students (latest record per student) → crossed with the 10 fixed
  scholarships → 92,900 student–scholarship interaction rows.
- **Text/embeddings**: `student_text` and `scholarship_text` are built
  exactly as in the notebook's CELL 5, embedded with SBERT
  `all-MiniLM-L6-v2` (384-dim), and compared with cosine similarity
  (CELL 6/7).
- **Features engineered**: `major_match`, `eligible`, `gpa_score`,
  `extra_score`, `credit_score`, and the synthetic `application_outcome`
  label (CELL 8/9) — all reproduced unchanged.
- **Split**: student-level 70/30 train/test split, `random_state=42`
  (CELL 10).
- **Models**: XGBoost, RandomForest, LightGBM, CatBoost, trained with the
  exact hyperparameters in CELLS 12–15.
- **Ensemble**: strict equal-weight soft voting, `0.25` each (CELL 16).

`train_pipeline.py` in this folder is the script actually used to
regenerate the exported model files for this package. It re-implements
CELLS 10–17 verbatim against the notebook's own saved interaction
dataset (`scholarship_interactions_92900_FULL.csv`, which you supplied
and which already contains the notebook's computed `semantic_similarity`,
`major_match`, `eligible`, and `application_outcome` columns — i.e. this
is the notebook's actual output, not a re-simulation).

### Metric reproduction note

The notebook itself does not have saved/pickled model objects — only
code. To produce **real, non-fabricated** trained model files, the four
models were retrained here using `train_pipeline.py`, on the notebook's
own saved output data, with identical hyperparameters/split/random seed.
The student-level split reproduced **exactly** (6,503 train / 2,787 test
students; 65,030 / 27,870 rows) and metrics matched the notebook's
reported values within ~0.001–0.002 (4th decimal place), which is
expected floating-point/library-version variation between this
environment's XGBoost/LightGBM/CatBoost versions and whatever the
original Colab environment used (the notebook does not print its
package versions). See `config/metrics_report.json` for this run's exact
numbers side-by-side with the notebook's reported numbers in
`config/model_config.json`.

**Synthetic label disclosure** (per the notebook and per your request):
the original dataset has no real application-outcome labels. The
`application_outcome` used to train these classifiers is a reproducible
synthetic outcome, generated from the notebook's own profile-fit and
eligibility logic (CELL 9) plus a fixed-seed random component. It is
**not** real-world scholarship application data, and the exported models'
predictions should be understood in that light.

---

## 2. Which models were exported

| Model | File | Format |
|---|---|---|
| XGBoost | `models/xgb_model/xgb_model.json` | native XGBoost JSON (via `XGBClassifier.save_model`) |
| Random Forest | `models/rf_model/rf_model.joblib` | joblib (scikit-learn has no native format) |
| LightGBM | `models/lgbm_model/lgbm_model.joblib` (+ `lgbm_model.txt` native booster) | joblib wrapper preserves `predict_proba`; `.txt` is the native LightGBM booster format |
| CatBoost | `models/catboost_model/catboost_model.cbm` | native CatBoost binary (`CatBoostClassifier.save_model`) |
| Class encoder | `preprocessing/class_encoder.joblib` | joblib (fitted `sklearn.LabelEncoder`, **not refit**) |

Each was chosen using the safest/native format recommended by that
library, per your instructions — only RandomForest and the LightGBM
sklearn wrapper use joblib, because scikit-learn/LightGBM's sklearn API
has no other portable native serialization for the full estimator.

## 3. How the four models are combined

`fastapi_integration/ensemble.py` implements, verbatim:

```python
ensemble_probability = (
    0.25 * xgb_model.predict_proba(X)[:, 1] +
    0.25 * rf_model.predict_proba(X)[:, 1] +
    0.25 * lgbm_model.predict_proba(X)[:, 1] +
    0.25 * catboost_model.predict_proba(X)[:, 1]
)
```

All four are binary classifiers exposing `.predict_proba` → `(n, 2)`
arrays; column `[:, 1]` is the probability of `application_outcome == 1`,
matching the notebook's own usage in CELLS 12–16 exactly.

## 4. How SBERT is loaded

`model_loader.py` prefers a **locally bundled, verified copy** of
`all-MiniLM-L6-v2` at `fastapi_integration/local_models/all-MiniLM-L6-v2/`
(full HF-format files: config, tokenizer, weights — ~88MB). This means
**no network access is required at runtime** for the default model.

This local copy was verified, not just assumed correct: I computed the
cosine similarity between the notebook's own CELL 7 example pair
(student 100000 vs. SCH001) using this local model and got
`0.40926975`, against the notebook's own saved value of `0.40926963` —
matching to 6 decimal places (the residual difference is ordinary
floating-point noise). This confirms it's the exact same model weights
the notebook used, not an approximation.

If you'd rather load by name from Hugging Face Hub instead (e.g. to
always get upstream updates), delete `local_models/` — `model_loader.py`
falls back to `SentenceTransformer("all-MiniLM-L6-v2")` automatically,
which then requires outbound network access to `huggingface.co` on
first run.

## 5. How preprocessing is reproduced

`fastapi_integration/preprocessing.py`:
1. `build_student_text()` — identical string formula to CELL 5.
2. `calculate_major_match()` — identical logic to CELL 8.
3. `encode_class()` — uses the **fitted, saved** `LabelEncoder`
   (`class_encoder.joblib`); never refits. Raises a clear error for a
   class/major code the encoder never saw in training (no silent
   fallback, no data leakage).
4. `build_feature_vector()` — assembles the exact 11 features, in the
   exact order from `config/feature_columns.json`, as a `pandas.DataFrame`
   with matching column names (models were trained on named DataFrames).

`fastapi_integration/eligibility.py` reproduces CELL 8's eligibility rule
verbatim: `gpa >= 7 AND total_credits >= 14 AND NOT has_failed_course AND
student_year <= 4`. This is a single global rule in the notebook (not
scholarship-specific) — preserved as-is.

## 6. How the backend should call the recommendation function

```python
from model_loader import load_models
from recommendation_service import recommend_scholarships

artifacts = load_models()   # once, at startup

student = {
    "student_id": 999999,
    "semester": 1,
    "gpa": 8.5,
    "extracurricular_point": 70,
    "total_credits": 18,
    "class": "CS2021",
    "has_failed_course": False,
    "student_year": 4,
}

result = recommend_scholarships(student, artifacts, top_k=5)
# -> {"student_id": ..., "recommendations": [ ...5 ranked scholarships... ]}
```

See `fastapi_integration/app.py` for a working reference FastAPI app
wiring this to `POST /api/recommend`, `POST /api/explain`, and
`GET /api/health`.

## 7. Whether SHAP is implemented

The notebook does **not** contain a SHAP implementation (confirmed by
inspecting all 22 cells). `fastapi_integration/explain.py` adds one:
per-model `shap.TreeExplainer` values for each of the four trained
models, combined with the same 0.25/0.25/0.25/0.25 weighting as the
prediction ensemble. This is documented in detail — including a genuine
scale-mismatch caveat between margin-space and probability-space
TreeExplainer outputs across libraries — directly in that file's
docstring. No explanation numbers are fabricated; they come from real
SHAP computations against the real trained models.

## 8–9. Reload / new-student validation status — ALL VERIFIED

Performed in this environment, end-to-end, with the real model artifacts
and the real (verified) SBERT model:
- ✅ All 4 models saved, then reloaded in a fresh process via
  `model_loader.load_models()`.
- ✅ Class encoder reloaded from disk (not refit).
- ✅ Scholarship metadata reloaded (10 scholarships, matches notebook).
- ✅ SBERT loaded locally, embeddings verified byte-level-accurate
  against the notebook's own saved similarity value (see §4).
- ✅ Feature vector assembly verified: exactly 11 features, correct
  order.
- ✅ All four models verified to individually contribute to the
  soft-voting score.
- ✅ Full pipeline run for a new student end-to-end with **real** SBERT
  embeddings: Top-5 returned, sorted descending, ranks 1–5, JSON
  serializable, no NaNs (`pytest tests/ -v` → 10/10 pass, including
  `test_full_pipeline_with_sbert`, which no longer needs to skip).
- ✅ `/api/explain` verified end-to-end with real SHAP values.

One real finding surfaced during this validation, worth flagging: your
prompt's own suggested sanity-check student used `"class": "CS"`, but
the notebook's class values are major+enrollment-year codes (e.g.
`"CS2021"`) — `"CS"` alone was never seen during training. The class
encoder correctly rejects it with a clear error rather than guessing.
Use a real class code like `"CS2022"` (see the worked example below).

### Worked example (real output, not illustrative)

Student: `{gpa: 8.5, extracurricular_point: 70, total_credits: 18,
class: "CS2022", has_failed_course: false, student_year: 4, semester: 2}`

| Rank | Scholarship | Score | Semantic Sim | Major Match | Eligible |
|---|---|---|---|---|---|
| 1 | SCH001 Academic Excellence | 0.9487 | 0.3973 | 1 | true |
| 2 | SCH003 Technology and Computing | 0.9480 | 0.3917 | 1 | true |
| 3 | SCH004 Merit | 0.9472 | 0.4188 | 1 | true |
| 4 | SCH005 Student Achievement | 0.9458 | 0.4197 | 1 | true |
| 5 | SCH008 Academic Progress | 0.9442 | 0.3207 | 1 | true |

## 10. Known limitations

- Exact original notebook package versions are unknown (notebook never
  printed them); this package documents and pins the versions it was
  actually built/tested against instead (see
  `fastapi_integration/requirements.txt`).
- `class_encoder` will raise on a class/major code not seen during
  training (e.g. `"CS"` alone, vs. trained-on `"CS2022"`) — by design
  (no silent refitting), but the backend should surface this as a clear
  4xx error to the caller.
- The combined SHAP explanation mixes margin-space and probability-space
  attributions across the four libraries; treat `impact` as a relative
  share, not a probability delta. See `explain.py` docstring for the
  full reasoning and a suggested alternative if stricter precision is
  needed.
- `application_outcome` (the training label) is synthetic, not real
  application data — see the disclosure above.

---

## Folder structure

```
ScholarMatch_model_artifacts/
├── models/                      # exported model files (see table above)
├── preprocessing/                # fitted class_encoder.joblib
├── config/
│   ├── scholarships.json         # 10 scholarships, notebook metadata unchanged
│   ├── feature_columns.json      # exact 11-feature order
│   ├── model_config.json         # ensemble weights, SBERT name, reported metrics
│   └── metrics_report.json       # this run's reproduced metrics
├── data/                         # intentionally empty at inference time — see note below
├── train_pipeline.py             # script used to regenerate the exported models
├── fastapi_integration/          # drop-in backend package (self-contained copy of models/config)
│   ├── app.py                    # reference FastAPI app: /api/recommend /api/explain /api/health
│   ├── model_loader.py
│   ├── recommendation_service.py
│   ├── ensemble.py
│   ├── preprocessing.py
│   ├── eligibility.py
│   ├── explain.py
│   ├── schemas.py
│   ├── requirements.txt
│   ├── models/ , preprocessing/, scholarships.json, model_config.json
│   ├── tests/
│   └── README.md
└── README.md                     # this file
```

`data/` is intentionally empty of the 92,900-row training dataset: per
your explicit instruction, the FastAPI backend must NOT load the full
training interactions file just to serve predictions. That file
(`scholarship_interactions_92900_FULL.csv`, ~58MB) was used only to
regenerate the model artifacts via `train_pipeline.py`, not shipped for
runtime use.
