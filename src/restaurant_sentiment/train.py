import os

from data_loader import load_and_preprocess_data
from joblib import dump

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

def train(filepath = "data/RestaurantReviews_HistoricDump.tsv", max_features=1420, test_size=0.20, random_state=0):
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
    print("Loading and preprocessing data...")
    X_raw, y = load_and_preprocess_data(filepath)

    # TODO: Optionally perform text vectorization in 'lib_ml' package
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

    # Predict on test data
    print("Predicting on test data...")
    y_pred = model.predict(X_test)

    # Evaluate model performance
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {acc}")

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

if __name__ == "__main__":
    train()
