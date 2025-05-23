import os
import requests
import argparse


def download_dataset(url, output_path):
    """
    Downloads the dataset at the specified URL. Expects a single file (e.g. CSV).

    Parameters:
    - url: str, URL to the dataset as a single file.
    - output_path: str, destination file to copy contents to.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Dataset downloaded successfully to: {output_path}")
    else:
        print(f"Failed to download dataset. Status code: {response.status_code}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="https://raw.githubusercontent.com/proksch/restaurant-sentiment/main/a1_RestaurantReviews_HistoricDump.tsv"
    )
    parser.add_argument(
        "--output",
        default="data/RestaurantReviews_HistoricDump.tsv"
    )

    args = parser.parse_args()
    download_dataset(args.url, args.output)