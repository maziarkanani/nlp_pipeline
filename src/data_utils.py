import pandas as pd
import spacy
from sklearn.model_selection import train_test_split

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def clean_text(text: str) -> str:
    doc = nlp(text.lower())
    tokens = [
        t.lemma_ for t in doc
        if not t.is_stop and not t.is_punct and t.is_alpha
    ]
    return " ".join(tokens)

def load_dataset(path: str = "data/dataset.csv", text_col="text", label_col="label"):
    df = pd.read_csv(path)
    df = df.dropna(subset=[text_col, label_col])

    df["clean_text"] = df[text_col].astype(str).apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df[label_col],
        test_size=0.2,
        random_state=42,
        stratify=df[label_col]
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_dataset()
    print("Train size:", len(X_train), "Test size:", len(X_test))
