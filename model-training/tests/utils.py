import os
from restaurant_sentiment.get_data import download_dataset
from restaurant_sentiment.preprocess import main

def load_sample_data(tmp_path):
    url = (
        "https://raw.githubusercontent.com/proksch/restaurant-sentiment/"
        "main/a1_RestaurantReviews_HistoricDump.tsv"
    )
    dataset_path = os.path.join(tmp_path, "data/RestaurantReviews_HistoricDump.tsv")
    download_dataset(url, dataset_path)
    main(
        data_path=os.path.join(tmp_path, "data"),
        filepath=dataset_path,
    )
    return tmp_path
