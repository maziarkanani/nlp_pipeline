from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from pathlib import Path

from data_utils import load_dataset

def train_tfidf_model():
    X_train, X_test, y_train, y_test = load_dataset()

    clf = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
            ("logreg", LogisticRegression(max_iter=1000))
        ]
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred)
    Path("reports").mkdir(exist_ok=True)
    with open("reports/metrics.txt", "w") as f:
        f.write(report)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(clf, "models/tfidf_logreg.joblib")

    print(report)
    print("Model saved to models/tfidf_logreg.joblib")

if __name__ == "__main__":
    train_tfidf_model()
