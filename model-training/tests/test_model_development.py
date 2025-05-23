import os
import pytest
from sklearn.metrics import accuracy_score
from joblib import load
from restaurant_sentiment.train import train
from restaurant_sentiment.get_data import download_dataset
from restaurant_sentiment.preprocess import main

@pytest.fixture
def model_and_data(tmp_path):
    URL = "https://raw.githubusercontent.com/proksch/restaurant-sentiment/main/a1_RestaurantReviews_HistoricDump.tsv"
    download_dataset(URL, os.path.join(tmp_path, "data/RestaurantReviews_HistoricDump.tsv"))
    main(data_path=os.path.join(tmp_path, 'data'), filepath=os.path.join(tmp_path, 'data/RestaurantReviews_HistoricDump.tsv'))
    train(data_path=os.path.join(tmp_path, "data"), model_path=os.path.join(tmp_path, "model"))
    model = load(os.path.join(tmp_path, "model/model.pkl"))
    bow = load(os.path.join(tmp_path, "model/bow.pkl"))
    X_test = load(os.path.join(tmp_path, "data/X_test.pkl"))
    y_test = load(os.path.join(tmp_path, "data/y_test.pkl"))
    return model, bow, X_test, y_test

@pytest.fixture
def data_slices():
    return {
        "positive_sentiment": lambda X, y: (X[y == 1], y[y == 1]),
        "negative_sentiment": lambda X, y: (X[y == 0], y[y == 0])
    }

def test_model_quality_on_slices(model_and_data, data_slices):
    # Belongs to case: Test model quality on important data slices
    model, bow, X_test, y_test = model_and_data

    for slice_name, slice_fn in data_slices.items():
        X_slice, y_slice = slice_fn(X_test, y_test)
        if len(X_slice) == 0:
            continue
        y_pred = model.predict(X_slice)
        accuracy = accuracy_score(y_slice, y_pred)

        assert accuracy > 0.5, f"Model accuracy on {slice_name} is below threshold: {accuracy}"