# 📘 Real-World NLP Classification Pipeline

A simple, end-to-end NLP pipeline for text classification, using TF-IDF, Logistic Regression, spaCy preprocessing, and SHAP explainability.
Includes an optional HuggingFace embedding variant.

---

## 📂 Project Structure

```
nlp_pipeline/
│
├── data/                     # dataset.csv (text,label)
├── models/                   # saved models
├── reports/                  # metrics + SHAP plots
│
├── src/
│   ├── data_utils.py         # loading + spaCy cleaning
│   ├── train_tfidf.py        # main TF-IDF classifier
│   ├── explain_model.py      # SHAP explanations
│   └── embed_hf.py           # optional HF embedding classifier
│
├── requirements.txt
└── README.md
```

---

## 📦 Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## 📥 Dataset Format

Place a CSV file here:

```
data/dataset.csv
```

With columns:

```
text,label
"This is an example sentence",positive
"Another text sample",negative
```

---

## 🚀 Training (TF-IDF + Logistic Regression)

```bash
python src/train_tfidf.py
```

Outputs:

* `models/tfidf_logreg.joblib`
* `reports/metrics.txt`

---

## 📊 SHAP Explainability

```bash
python src/explain_model.py
```

Outputs:

* `reports/shap_summary.png`

This shows which words push predictions toward each class.

---

## 🤗 Optional: HuggingFace Embedding Classifier

Uses `sentence-transformers/all-MiniLM-L6-v2` to create embeddings.

```bash
python src/embed_hf.py
```

Outputs:

* `models/hf_logreg.joblib`
* `reports/hf_metrics.txt`
