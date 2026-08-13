This folder is intentionally empty at delivery time.

Per the project requirements, the FastAPI backend must NOT load the
92,900-row training interaction dataset to serve predictions — only
trained models + preprocessing artifacts + SBERT + scholarship
metadata + feature config are needed at inference time (see
model_loader.py).

If you need to regenerate the models (e.g. after new training data),
place `scholarship_interactions_92900_FULL.csv` here and run
`../train_pipeline.py` from this directory's parent.
