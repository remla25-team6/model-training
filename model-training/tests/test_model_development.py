import os
import pytest
from sklearn.metrics import accuracy_score
from joblib import load
from restaurant_sentiment.train import train
from tests.utils import load_sample_data
import numpy as np
import tempfile
import pandas as pd
from restaurant_sentiment.preprocess import load_and_preprocess_data


@pytest.fixture
def load_data(tmp_path):
    return load_sample_data(tmp_path)


@pytest.fixture
def data_slices():
    return {
        "positive_sentiment": lambda X, y: (X[y == 1], y[y == 1]),
        "negative_sentiment": lambda X, y: (X[y == 0], y[y == 0]),
    }


def test_model_1_model_quality_on_slices(load_data, data_slices):
    # Belongs to case: Test model quality on important data slices
    tmp_path = load_data
    train(
        data_path=os.path.join(tmp_path, "data"),
        model_path=os.path.join(tmp_path, "model"),
    )
    model = load(os.path.join(tmp_path, "model/model.pkl"))
    X_test = load(os.path.join(tmp_path, "data/X_test.pkl"))
    y_test = load(os.path.join(tmp_path, "data/y_test.pkl"))

    for slice_name, slice_fn in data_slices.items():
        X_slice, y_slice = slice_fn(X_test, y_test)
        if len(X_slice) == 0:
            continue
        y_pred = model.predict(X_slice)
        accuracy = accuracy_score(y_slice, y_pred)

        assert (
            accuracy > 0.5
        ), f"Model accuracy on {slice_name} is below threshold: {accuracy}"


def test_model_2_non_determinism_robustness(load_data, data_slices, tolerance=0.05):
    tmp_path = load_data
    accuracies = {name: [] for name in data_slices}

    for seed in [0, 1, 2]:
        model_path = os.path.join(tmp_path, f"model{seed}")
        train(
            data_path=os.path.join(tmp_path, "data"),
            model_path=model_path,
        )
        model = load(os.path.join(model_path, "model.pkl"))
        X_test = load(os.path.join(tmp_path, "data/X_test.pkl"))
        y_test = load(os.path.join(tmp_path, "data/y_test.pkl"))

        slice_accuracies = evaluate_model_on_slices(model, X_test, y_test, data_slices)
        for slice_name, acc in slice_accuracies.items():
            accuracies[slice_name].append(acc)

    for slice_name, accs in accuracies.items():
        mean_acc = sum(accs) / len(accs)
        for acc in accs:
            assert (
                abs(acc - mean_acc) <= tolerance
            ), (
                f"Accuracy {acc:.3f} deviates from mean accuracy {mean_acc:.3f} "
                f"by more than {tolerance:.3f} in slice: {slice_name}"
            )

def evaluate_model_on_slices(model, X_test, y_test, data_slices):
    slice_accuracies = {}
    for slice_name, slice_fn in data_slices.items():
        X_slice, y_slice = slice_fn(X_test, y_test)
        y_pred = model.predict(X_slice)
        accuracy = accuracy_score(y_slice, y_pred)
        slice_accuracies[slice_name] = accuracy
    return slice_accuracies

def test_model_3_permutation_invariance(load_data):
    """
    Test that shuffling the test data does not affect the model's predictions.
    """
    tmp_path = load_data
    train(
        data_path=os.path.join(tmp_path, "data"),
        model_path=os.path.join(tmp_path, "model"),
    )
    model = load(os.path.join(tmp_path, "model/model.pkl"))
    X_test = load(os.path.join(tmp_path, "data/X_test.pkl"))
    y_test = load(os.path.join(tmp_path, "data/y_test.pkl"))

    # Get predictions for original order
    y_pred_orig = model.predict(X_test)

    # Shuffle X_test and y_test in the same way
    idx = np.arange(len(X_test))
    np.random.shuffle(idx)
    X_shuffled = X_test[idx]
    y_shuffled = y_test[idx]
    y_pred_shuffled = model.predict(X_shuffled)

    # Unshuffle predictions to original order
    unshuffled_pred = np.empty_like(y_pred_shuffled)
    unshuffled_pred[idx] = y_pred_shuffled

    assert (y_pred_orig == unshuffled_pred).all(), "Predictions changed after shuffling test data."
