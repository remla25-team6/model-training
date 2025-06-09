import os
import argparse
import json
from sklearn.metrics import confusion_matrix, accuracy_score
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

    with open(os.path.join(model_path, "metrics.json"), "w") as f:
        json.dump({"test_accuracy": acc}, f)

    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--model", default="model")
    args = parser.parse_args()

    evaluate(data_path=args.data, model_path=args.model)
