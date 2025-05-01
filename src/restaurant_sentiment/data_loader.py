import re
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords from nltk
nltk.download('stopwords')

def load_and_preprocess_data(filepath, num_reviews=900):
    """
    Loads and preprocesses the dataset.
    
    Parameters:
    - filepath: str, path to the dataset (CSV file).
    - num_reviews: int, the number of reviews to process (default is 900).
    
    Returns:
    - tuple: 
      - A list of preprocessed reviews (corpus).
      - A numpy array of the labels from the dataset.
    """
    print(f"Loading dataset from {filepath}...")
    
    try:
        # Load dataset from the given filepath, considering only the first 'num_reviews' rows
        dataset = pd.read_csv(filepath, delimiter='\t', quoting=3)
        dataset = dataset.iloc[:num_reviews, :]
    except Exception as e:
        print(f"Error loading dataset from {filepath}: {e}")

    print("Loading complete.")
    
    try:
        print(f"Preprocessing reviews...")

        # Preprocess reviews
        # TODO: Replace with call to 'lib_ml' once package is available
        corpus = _preprocess(dataset, num_reviews)
    except Exception as e:
        print(f"Error preprocessing reviews: {e}")

    print("Preprocessing complete.")

    # Return the preprocessed reviews and labels
    return corpus, dataset.iloc[:, -1].values 

def _preprocess(dataset, num_reviews):
    """
    Preprocesses the reviews.
    
    Parameters:
    - dataset: pandas DataFrame, the dataset containing reviews.
    - num_reviews: int, the number of reviews to process.
    
    Returns:
    - list: A list of preprocessed reviews (corpus).
    """
    # Initialize the Porter Stemmer and stopwords
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    
    # Remove 'not' from the stopwords list to retain negatives
    all_stopwords.remove('not')

    # Initialize list to store the preprocessed reviews
    corpus = [] 

    # Process each review 
    for i in range(num_reviews):
        # Remove non-alphabetical characters and split the review into words
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()

        # Apply stemming and remove stopwords
        review = [ps.stem(word) for word in review if word not in set(all_stopwords)]

        # Join the processed words back into a single string and append to corpus
        review = ' '.join(review)
        corpus.append(review)

    return corpus
