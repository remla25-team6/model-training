import os
import pytest
import joblib
import pandas as pd
from tests.utils import load_sample_data
from restaurant_sentiment.train import train

REPAIR_PATH = "data/repair_dataset.tsv"

@pytest.fixture
def load_data(tmp_path):
    return load_sample_data(tmp_path)

def append_to_repair_file(text, label):
    """
    Append a single (text, label) pair to the repair dataset. 
    This allows failed metamorphic test cases to be captured for future retraining
    for automatic inconsistency repair.
    """
    # Used ChatGPT 4o to append data to repair files
    os.makedirs(os.path.dirname(REPAIR_PATH), exist_ok=True)
    row = pd.DataFrame([[text, label]], columns=["Review", "Sentiment"])
    row.to_csv(
        REPAIR_PATH,
        sep="\t",
        index=False,
        mode="a",
        header=not os.path.exists(REPAIR_PATH))

def test_model_1_synonym_invariance(load_data):
    # Synonymous inputs should result in same prediction
    tmp_path = load_data
    model_path = os.path.join(tmp_path, "model")
    data_path = os.path.join(tmp_path, "data")

    train(data_path=data_path, model_path=model_path)

    model = joblib.load(os.path.join(model_path, "model.pkl"))
    bow = joblib.load(os.path.join(model_path, "bow.pkl"))

    synonym_pairs_with_labels = [
        ("The food was great", "The food was good", 1),
        ("The people were horrible", "The people were terrible", 0),
        ("I liked the game", "I loved the game", 1),
        ("This house is bad", "This house is gross", 0),
    ]

    for s1, s2, label in synonym_pairs_with_labels:
        x1 = bow.transform([s1]).toarray()
        x2 = bow.transform([s2]).toarray()
        pred1 = model.predict(x1)[0]
        pred2 = model.predict(x2)[0]

        if pred1 != pred2:
            # Save variants to repair file for automatic inconsistency repair
            append_to_repair_file(s1, label)
            append_to_repair_file(s2, label)

        assert pred1 == pred2, (
            f"Model gave different predictions for synonyms:\n"
            f"'{s1}' → {pred1}, '{s2}' → {pred2}"
        )
