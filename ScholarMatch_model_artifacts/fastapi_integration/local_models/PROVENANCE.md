This is a full local copy of sentence-transformers/all-MiniLM-L6-v2
(config, tokenizer, and weights — pytorch_model.bin), the same model
named in the training notebook.

Verification performed before bundling: embedding the exact CELL 5
student/scholarship text pair from the notebook's own first row
(student 100000 vs. SCH001) with this local copy and computing cosine
similarity reproduced the notebook's own saved semantic_similarity
value to 6 decimal places (0.40926975 here vs. 0.40926963 in
scholarship_interactions_92900_FULL.csv), confirming this is the same
model weights, not an approximation.

Bundled so the backend needs zero outbound network access at runtime
to compute semantic_similarity for new students.
