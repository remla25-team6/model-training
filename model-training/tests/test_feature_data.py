import os
import pytest
import pandas as pd
from sklearn.metrics import accuracy_score
from joblib import load

from restaurant_sentiment.preprocess import load_and_preprocess_data
from restaurant_sentiment.train import train
from tests.utils import load_sample_data


@pytest.fixture
def load_data(tmp_path):
    return load_sample_data(tmp_path)

@pytest.fixture
def sample_data(tmp_path):
    file_path = tmp_path / "test_data.tsv"
    data = pd.DataFrame(
        {"Review": ["Good and food", "Bad service"], "Sentiment": [1, 0]}
    )
    data.to_csv(file_path, sep="\t", index=False)
    return file_path


def test_data_1_load_and_preprocess_data(sample_data):
    corpus, labels = load_and_preprocess_data(sample_data)
    assert len(corpus) == 2, f"corpus length is not 2 but {len(corpus)}"
    assert len(labels) == 2, f"labels length is not 2 but {len(labels)}"


def test_data_2_stopword_removal(sample_data):
    # Belongs to case: Test all code that creates input features
    corpus, _ = load_and_preprocess_data(sample_data)
    assert "and" not in corpus[0].lower().split()

def test_data_3_cost_features(load_data):
    # The cost of features is being tested.
    # This test justifies using the number of features we use.
    # If using less features results in better performance,
    # measured by a 10% increase in performance for statistical significance,
    # then this test case fails.
    tmp_path = load_data
    max_feature_opts = [1000, 1420]
    accuracies = []
    for max_features in max_feature_opts:
        train(
            data_path=os.path.join(tmp_path, "data"),
            model_path=os.path.join(tmp_path, "model"),
            max_features=max_features
        )
        model = load(os.path.join(tmp_path, "model/model.pkl"))
        X_test = load(os.path.join(tmp_path, "data/X_test.pkl"))
        y_test = load(os.path.join(tmp_path, "data/y_test.pkl"))

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        accuracies.append(accuracy)

    assert (
        accuracies[0] / accuracies[1] < 1.1
    ), (
        f"Accuracy with {max_feature_opts[0]} features is more than "
        f"10% larger than with {max_feature_opts[1]} features."
    )
