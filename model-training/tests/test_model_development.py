import os
import pytest
from sklearn.metrics import accuracy_score
from joblib import load
from restaurant_sentiment.train import train
from restaurant_sentiment.get_data import download_dataset
from restaurant_sentiment.preprocess import main


@pytest.fixture
def load_data(tmp_path):
    url = "https://raw.githubusercontent.com/proksch/restaurant-sentiment/" \
    "main/a1_RestaurantReviews_HistoricDump.tsv"
    download_dataset(
        url, os.path.join(tmp_path, "data/RestaurantReviews_HistoricDump.tsv")
    )
    main(
        data_path=os.path.join(tmp_path, "data"),
        filepath=os.path.join(tmp_path, "data/RestaurantReviews_HistoricDump.tsv"),
    )
    return tmp_path


@pytest.fixture
def data_slices():
    return {
        "positive_sentiment": lambda X, y: (X[y == 1], y[y == 1]),
        "negative_sentiment": lambda X, y: (X[y == 0], y[y == 0]),
    }


def test_model_6_model_quality_on_slices(load_data, data_slices):
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

def test_non_determinism_robustness(load_data, data_slices, tolerance = 0.05):
    # Test for non-determinism robustness exist and use data slices to test model capabilities.
    tmp_path = load_data
    random_states = [0,1,2]
    accuracies = {slice_name: [] for slice_name, slice_fn in data_slices.items()}
    for random_state in random_states:
        train(
            data_path=os.path.join(tmp_path, "data"),
            model_path=os.path.join(tmp_path, f"model{random_state}"),
        )
        model = load(os.path.join(tmp_path, f"model{random_state}/model.pkl"))
        X_test = load(os.path.join(tmp_path, "data/X_test.pkl"))
        y_test = load(os.path.join(tmp_path, "data/y_test.pkl"))
        for slice_name, slice_fn in data_slices.items():
            X_slice, y_slice = slice_fn(X_test, y_test)
            y_pred = model.predict(X_slice)
            accuracy = accuracy_score(y_slice, y_pred)
            accuracies[slice_name].append(accuracy)
    for slice_name, _ in data_slices.items():
        mean_acc = sum(accuracies[slice_name]) / len(accuracies[slice_name])
        for acc in accuracies[slice_name]:
            assert (
                abs(acc - mean_acc) <= tolerance
            ), f"Accuracy {acc} deviates from mean accuracy {mean_acc} by more than {tolerance} in slice: {slice_name}"