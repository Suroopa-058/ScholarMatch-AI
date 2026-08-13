"""
Reproduces CELLS 10-17 of the original ScholarMatch notebook exactly,
using the notebook's own saved interaction dataset
(scholarship_interactions_92900_FULL.csv) as the source of truth.

This does NOT redesign anything. Same split logic, same encoder,
same feature list/order, same hyperparameters, same ensemble weights.
"""
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    ndcg_score,
)
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib

RANDOM_STATE = 42

interaction_df = pd.read_csv("interactions.csv")
interaction_df["has_failed_course"] = interaction_df["has_failed_course"].astype(bool)
interaction_df["eligible"] = interaction_df["eligible"].astype(bool)

# ------------------------------------------------------------
# CELL 10 — student-level train/test split (identical logic)
# ------------------------------------------------------------
unique_students = interaction_df["student_id"].unique()
train_students, test_students = train_test_split(
    unique_students, test_size=0.30, random_state=RANDOM_STATE
)
train_students = set(train_students)
test_students = set(test_students)

train_rec = interaction_df[interaction_df["student_id"].isin(train_students)].copy()
test_rec = interaction_df[interaction_df["student_id"].isin(test_students)].copy()

print("Training students:", len(train_students))
print("Testing students:", len(test_students))
print("Training interaction rows:", len(train_rec))
print("Testing interaction rows:", len(test_rec))

# ------------------------------------------------------------
# CELL 11 — class encoder fit on FULL interaction_df (as notebook does)
# ------------------------------------------------------------
class_encoder = LabelEncoder()
class_encoder.fit(interaction_df["class"].astype(str))

train_rec["class_encoded"] = class_encoder.transform(train_rec["class"].astype(str))
test_rec["class_encoded"] = class_encoder.transform(test_rec["class"].astype(str))

FEATURES = [
    "gpa", "extracurricular_point", "total_credits", "has_failed_course",
    "student_year", "semester", "class_encoded", "academic_weight",
    "extracurricular_weight", "major_match", "semantic_similarity",
]
TARGET = "application_outcome"

X_train = train_rec[FEATURES].copy()
X_test = test_rec[FEATURES].copy()
y_train = train_rec[TARGET].copy()
y_test = test_rec[TARGET].copy()

X_train["has_failed_course"] = X_train["has_failed_course"].astype(int)
X_test["has_failed_course"] = X_test["has_failed_course"].astype(int)

print("Training feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)

# ------------------------------------------------------------
# CELL 12 — XGBoost
# ------------------------------------------------------------
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()
scale_pos_weight = negative_count / positive_count

xgb_clf = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective="binary:logistic", eval_metric="logloss",
    scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, n_jobs=-1,
)
xgb_clf.fit(X_train, y_train)
xgb_prob = xgb_clf.predict_proba(X_test)[:, 1]
xgb_pred = xgb_clf.predict(X_test)

# ------------------------------------------------------------
# CELL 13 — Random Forest
# ------------------------------------------------------------
rf_clf = RandomForestClassifier(
    n_estimators=300, max_depth=12, min_samples_split=5, min_samples_leaf=2,
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
)
rf_clf.fit(X_train, y_train)
rf_prob = rf_clf.predict_proba(X_test)[:, 1]
rf_pred = rf_clf.predict(X_test)

# ------------------------------------------------------------
# CELL 14 — LightGBM
# ------------------------------------------------------------
lgbm_clf = LGBMClassifier(
    n_estimators=300, max_depth=8, learning_rate=0.05, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
)
lgbm_clf.fit(X_train, y_train)
lgbm_prob = lgbm_clf.predict_proba(X_test)[:, 1]
lgbm_pred = lgbm_clf.predict(X_test)

# ------------------------------------------------------------
# CELL 15 — CatBoost
# ------------------------------------------------------------
catboost_clf = CatBoostClassifier(
    iterations=300, depth=7, learning_rate=0.05,
    loss_function="Logloss", eval_metric="AUC",
    auto_class_weights="Balanced", random_seed=RANDOM_STATE, verbose=False,
)
catboost_clf.fit(X_train, y_train)
catboost_prob = catboost_clf.predict_proba(X_test)[:, 1]
catboost_pred = catboost_clf.predict(X_test).astype(int).ravel()


def evaluate(name, y_true, pred, prob):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, prob),
        "average_precision": average_precision_score(y_true, prob),
    }


results = [
    evaluate("xgboost", y_test, xgb_pred, xgb_prob),
    evaluate("random_forest", y_test, rf_pred, rf_prob),
    evaluate("lightgbm", y_test, lgbm_pred, lgbm_prob),
    evaluate("catboost", y_test, catboost_pred, catboost_prob),
]

# ------------------------------------------------------------
# CELL 16 — equal weight soft voting ensemble
# ------------------------------------------------------------
ensemble_probability = 0.25 * xgb_prob + 0.25 * rf_prob + 0.25 * lgbm_prob + 0.25 * catboost_prob
ensemble_prediction = (ensemble_probability >= 0.50).astype(int)
results.append(evaluate("ensemble", y_test, ensemble_prediction, ensemble_probability))

print(json.dumps(results, indent=2))

# ------------------------------------------------------------
# CELL 17 — ranking metrics
# ------------------------------------------------------------
ranking_df = test_rec[["student_id", "scholarship_id", "application_outcome"]].copy()
ranking_df["ensemble_probability"] = ensemble_probability

K_VALUES = [1, 3, 5]
precision_results, recall_results, ndcg_results = {}, {}, {}
for k in K_VALUES:
    precision_scores, recall_scores, ndcg_scores_list = [], [], []
    for student_id, group in ranking_df.groupby("student_id"):
        group = group.sort_values("ensemble_probability", ascending=False)
        top_k = group.head(k)
        actual_positive = group["application_outcome"].sum()
        recommended_positive = top_k["application_outcome"].sum()
        precision_scores.append(recommended_positive / k)
        recall_scores.append(recommended_positive / actual_positive if actual_positive > 0 else 0.0)
        y_true = group["application_outcome"].values.reshape(1, -1)
        y_score = group["ensemble_probability"].values.reshape(1, -1)
        ndcg_scores_list.append(ndcg_score(y_true, y_score, k=k))
    precision_results[k] = float(np.mean(precision_scores))
    recall_results[k] = float(np.mean(recall_scores))
    ndcg_results[k] = float(np.mean(ndcg_scores_list))

ranking_metrics = {"precision_at_k": precision_results, "recall_at_k": recall_results, "ndcg_at_k": ndcg_results}
print(json.dumps(ranking_metrics, indent=2))

# ------------------------------------------------------------
# Save everything needed downstream
# ------------------------------------------------------------
joblib.dump(class_encoder, "class_encoder.joblib")
xgb_clf.save_model("xgb_model.json")
joblib.dump(rf_clf, "rf_model.joblib")
lgbm_clf.booster_.save_model("lgbm_model.txt")
joblib.dump(lgbm_clf, "lgbm_model.joblib")
catboost_clf.save_model("catboost_model.cbm")

with open("metrics_report.json", "w") as f:
    json.dump({"model_metrics": results, "ranking_metrics": ranking_metrics}, f, indent=2)

# Save X_test sample + test predictions for later validation
X_test.head(20).to_csv("X_test_sample.csv", index=False)

print("DONE")
