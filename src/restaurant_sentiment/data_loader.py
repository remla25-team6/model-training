import re
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords from nltk
nltk.download('stopwords')

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
        corpus = _preprocess(dataset)
    except Exception as e:
        print(f"Error preprocessing reviews: {e}")
        return [], []

    print("Preprocessing complete.")
    return corpus, dataset.iloc[:, -1].values 

def _preprocess(dataset):
    """
    Preprocesses the reviews in the dataset.
    
    Parameters:
    - dataset: pandas DataFrame, the dataset containing reviews.
    
    Returns:
    - list: A list of preprocessed reviews (corpus).
    """
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    if 'not' in all_stopwords:
        all_stopwords.remove('not')

    corpus = []
    for review in dataset['Review']:
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower().split()
        review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
        corpus.append(' '.join(review))

    return corpus
