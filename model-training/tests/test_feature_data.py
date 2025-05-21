import pytest
import pandas as pd
from restaurant_sentiment.preprocess import load_and_preprocess_data

@pytest.fixture
def sample_data(tmp_path):
    file_path = tmp_path / "test_data.tsv"
    data = pd.DataFrame({
        'Review': ['Good and food', 'Bad service'],
        'Sentiment': [1, 0]
    })
    data.to_csv(file_path, sep='\t', index=False)
    return file_path

def test_load_and_preprocess_data(sample_data):
    corpus, labels = load_and_preprocess_data(sample_data)
    assert len(corpus) == 2
    assert len(labels) == 2

def test_stopword_removal(sample_data):
    # Belongs to case: Test all code that creates input features
    corpus, _ = load_and_preprocess_data(sample_data)
    assert "and" not in corpus[0].lower().split()