import os
import pytest
from joblib import load
from sklearn.metrics import accuracy_score
from restaurant_sentiment.train import train
from restaurant_sentiment.get_data import download_dataset
from restaurant_sentiment.preprocess import main

DATA_URL = "https://raw.githubusercontent.com/proksch/restaurant-sentiment/main/a1_RestaurantReviews_HistoricDump.tsv"

@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    # 1. Download
    data_dir = tmp_path_factory.mktemp("data")
    raw_tsv = data_dir / "RestaurantReviews_HistoricDump.tsv"
    download_dataset(DATA_URL, raw_tsv)
    # 2. Preprocess
    main(data_path=str(data_dir), filepath=str(raw_tsv))
    return str(data_dir)

@pytest.fixture(scope="session")
def baseline_accuracy(data_dir, tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("model0")
    # train with the default (seed=0)
    train(data_path=data_dir, model_path=str(model_dir), random_state=0)
    model = load(os.path.join(model_dir, "model.pkl"))
    X_test = load(os.path.join(data_dir, "X_test.pkl"))
    y_test = load(os.path.join(data_dir, "y_test.pkl"))
    return accuracy_score(y_test, model.predict(X_test))

@pytest.mark.parametrize("seed", [1, 2, 3])
def test_accuracy_stability(data_dir, baseline_accuracy, tmp_path, seed):
    model_dir = tmp_path / f"model_seed_{seed}"
    train(data_path=data_dir, model_path=str(model_dir), random_state=seed)
    model = load(os.path.join(model_dir, "model.pkl"))
    X_test = load(os.path.join(data_dir, "X_test.pkl"))
    y_test = load(os.path.join(data_dir, "y_test.pkl"))
    acc = accuracy_score(y_test, model.predict(X_test))
    # allow up to 8% relative difference
    assert acc == pytest.approx(baseline_accuracy, abs=0.08), (
        f"Seed {seed}: accuracy {acc} differs from baseline {baseline_accuracy}"
    )
