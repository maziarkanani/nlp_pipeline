import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_dataset
from pathlib import Path

def explain_model(num_samples: int = 200):
    X_train, X_test, y_train, y_test = load_dataset()

    model = joblib.load("models/tfidf_logreg.joblib")
    # model = Pipeline(tfidf, logreg)
    vectorizer = model.named_steps["tfidf"]
    logreg = model.named_steps["logreg"]

    # Take a subset for SHAP (linear model -> fast)
    X_sample = X_test.sample(min(num_samples, len(X_test)), random_state=0)
    X_vec = vectorizer.transform(X_sample)

    explainer = shap.LinearExplainer(logreg, X_vec, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_vec)

    # Get feature names
    feature_names = np.array(vectorizer.get_feature_names_out())

    Path("reports").mkdir(exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_vec,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    plt.savefig("reports/shap_summary.png", dpi=200)
    plt.close()

    print("Saved SHAP summary plot to reports/shap_summary.png")

if __name__ == "__main__":
    explain_model()
