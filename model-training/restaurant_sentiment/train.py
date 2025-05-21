# Disclaimer: Documentation was refined using ChatGPT 4o
import os
import argparse

from joblib import load, dump

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def train(data_path = "data", max_features=1420, test_size=0.20, random_state=0):
    """
    Loads data, vectorizes text using Bag-of-Words, and trains a Naive Bayes classifier.
    
    Parameters:
    - filepath: str, path to the TSV dataset file.
    - max_features: int, number of features for CountVectorizer (default 1420).
    - test_size: float, proportion of dataset to include in test split (default 0.20).
    - random_state: int, random seed for reproducibility (default 0).
    
    Returns:
    - None
    """
    print("Loading data...")
    X_raw, y = load(os.path.join(data_path, 'corpus.pkl')), load(os.path.join(data_path, 'labels.pkl'))

    # Convert text data to feature vectors using CountVectorizer
    print("Vectorizing text...")
    cv = CountVectorizer(max_features=max_features)
    X = cv.fit_transform(X_raw).toarray()

    # Split data into training and testing sets
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train the Gaussian Naive Bayes classifier
    print("Training Naive Bayes classifier...")
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Create 'model' directory if it does not exist
    os.makedirs("model", exist_ok=True)

    # Save BoW dictionary to use in inference
    print("Saving BoW dictionary to .pkl file...")
    dump(cv, "model/bow.pkl")
    print("Completed BoW dictionary saving.")

    # Save model
    print("Saving model to .pkl file...")
    dump(model, "model/model.pkl")
    print("Completed model saving.")

    # Dump the train and test splits for the evaluation step
    os.makedirs(data_path, exist_ok=True)

    dump(X_train, os.path.join(data_path, 'X_train.pkl'))
    dump(X_test, os.path.join(data_path, 'X_test.pkl'))
    dump(y_train, os.path.join(data_path, 'y_train.pkl'))
    dump(y_test, os.path.join(data_path, 'y_test.pkl'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data"
    )
    args = parser.parse_args()

    train(data_path=args.data)
