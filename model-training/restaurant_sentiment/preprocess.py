# Disclaimer: Documentation was refined using ChatGPT 4o
import os
import argparse

import pandas as pd
from joblib import dump

from lib_ml.preprocess import preprocess


def load_and_preprocess_data(filepath, num_reviews=None):
    """
    Loads and preprocesses the dataset.

    Parameters:
    - filepath: str, path to the dataset in a TSV file.
    - num_reviews: int or None, the number of reviews to process (process all if None).

    Returns:
    - tuple:
      - A list of preprocessed reviews (corpus).
      - A numpy array of the labels from the dataset.
    """
    print(f"Loading dataset from {filepath}...")

    try:
        dataset = pd.read_csv(filepath, delimiter="\t", quoting=3)
        if num_reviews is not None:
            dataset = dataset.iloc[:num_reviews, :]
    except Exception as e:
        print(f"Error loading dataset from {filepath}: {e}")
        return [], []

    print("Loading complete.")

    try:
        print("Preprocessing reviews...")
        corpus = preprocess(dataset, dataset.shape[0])
    except Exception as e:
        print(f"Error preprocessing reviews: {e}")
        return [], []

    print("Preprocessing complete.")
    return corpus, dataset.iloc[:, -1].to_numpy()


def main(data_path="data", filepath="data/RestaurantReviews_HistoricDump.tsv"):
    """
    Main function to preprocess the dataset and store the corpus and labels to disk.

    Parameters:
    - data_path: str, path to training data.
    - filepath: str, path to the dataset in a TSV file.

    Returns:
    - None
    """
    os.makedirs(data_path, exist_ok=True)

    corpus, labels = load_and_preprocess_data(filepath)

    corpus_file = os.path.join(data_path, "corpus.pkl")
    labels_file = os.path.join(data_path, "labels.pkl")

    dump(corpus, corpus_file)
    dump(labels, labels_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--filepath", default="data/RestaurantReviews_HistoricDump.tsv")
    args = parser.parse_args()
    main(data_path=args.data, filepath=args.filepath)
