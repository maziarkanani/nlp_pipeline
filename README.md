Title: “Real-World NLP Classification Pipeline”

Problem: e.g. fake-news detection / abstract classification

Methods: spaCy preprocessing, TF-IDF + Logistic Regression, optional HF embeddings

Explainability: SHAP summary plot

How to run:

put data/dataset.csv with text,label

run train_tfidf.py

run explain_model.py

optional embed_hf.py

Results: a short example of metrics + SHAP screenshot
