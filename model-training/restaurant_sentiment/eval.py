import os
import argparse
import json
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score
)
from joblib import load


def evaluate(data_path="data", model_path="model"):
    X_test = load(os.path.join(data_path, "X_test.pkl"))
    y_test = load(os.path.join(data_path, "y_test.pkl"))
    model = load(os.path.join(model_path, "model.pkl"))
    # Predict on test data
    print("Predicting on test data...")
    y_pred = model.predict(X_test)

    # Evaluate model performance
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(y_test, y_pred)
    n_samples = len(y_test)

    metrics = {
        "test_accuracy": acc,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1_score": f1,
        "test_cohens_kappa": kappa,
        "test_samples": n_samples,
        "confusion_matrix": cm.tolist()
    }

    # Dump to json
    with open(os.path.join(model_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {acc}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Test Samples: {n_samples}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--model", default="model")
    args = parser.parse_args()

    evaluate(data_path=args.data, model_path=args.model)
