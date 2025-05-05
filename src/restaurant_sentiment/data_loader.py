import pandas as pd

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
        dataset = pd.read_csv(filepath, delimiter='\t', quoting=3)
        if num_reviews is not None:
            dataset = dataset.iloc[:num_reviews, :]
    except Exception as e:
        print(f"Error loading dataset from {filepath}: {e}")
        return [], []

    print("Loading complete.")
    
    try:
        print(f"Preprocessing reviews...")
        corpus = preprocess(dataset, dataset.shape[0])
    except Exception as e:
        print(f"Error preprocessing reviews: {e}")
        return [], []

    print("Preprocessing complete.")
    return corpus, dataset.iloc[:, -1].values 