import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def embed_texts(texts, tokenizer, model, batch_size=32, device="cpu"):
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc)
            # mean pooling
            emb = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0).numpy()

def train_hf_classifier(path="data/dataset.csv", text_col="text", label_col="label"):
    df = pd.read_csv(path).dropna(subset=[text_col, label_col])

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col].astype(str),
        df[label_col],
        test_size=0.2,
        random_state=42,
        stratify=df[label_col]
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print("Embedding train texts...")
    X_train_emb = embed_texts(X_train, tokenizer, model, device=device)
    print("Embedding test texts...")
    X_test_emb = embed_texts(X_test, tokenizer, model, device=device)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_emb, y_train)
    y_pred = clf.predict(X_test_emb)

    report = classification_report(y_test, y_pred)
    Path("reports").mkdir(exist_ok=True)
    with open("reports/hf_metrics.txt", "w") as f:
        f.write(report)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(clf, "models/hf_logreg.joblib")

    print(report)
    print("Saved HF classifier and metrics.")

if __name__ == "__main__":
    train_hf_classifier()
