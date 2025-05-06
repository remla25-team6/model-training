# model-training
This repository contains the machine learning training pipeline for a sentiment classification model based on restaurant reviews. It is inspired by the GitHub repository: [proksch/restaurant-sentiment](https://github.com/proksch/restaurant-sentiment).

The training pipeline performs the following steps:

1. Loads a labelled dataset in `.tsv` format containing restaurant reviews.
2. Preprocesses the data using methods from the `lib-ml` package.
3. Trains a Naive Bayes classifier.
4. Saves the trained model locally to `model/model.pkl`.
5. Publishes a versioned model artifact to **GitHub Releases**.


## Training Data
The dataset is located in the `/data` folder in `.tsv` format and is sourced from:
🔗 [https://github.com/proksch/restaurant-sentiment](https://github.com/proksch/restaurant-sentiment)


## Local Setup
**Requirements:**
* Python `3.12.3`
* `pip`

**Setup Virtual Environment:**
Run the following command from the project root:
```bash
python -m venv <venv_name>
source <venv_name>/bin/activate  # For Unix/macOS
# Or use <venv_name>\Scripts\activate on Windows

pip install -r requirements.txt
```

To deactive after use:
```bash
deactivate
```


**Run Code (i.e. Train Model):**
Run the following command from the project root:
```bash
python src/restaurant_sentiment/train.py
```


## Release Trained Model
To publish a trained model to GitHub Releases, create a version tag using semantic versioning:
```bash
git tag v<MAJOR>.<MINOR>.<PATCH>
git push origin v<MAJOR>.<MINOR>.<PATCH>
```

This will trigger the `release.yml` GitHub Actions workflow, which trains the model and uploads it as an artifact to GitHub Releases. Once published, model releases are publicly accessible. For example, the download link for version `v0.1.0` would be:
[https://github.com/remla25-team6/model-training/releases/download/v0.1.0/model-v0.1.0.pkl](https://github.com/remla25-team6/model-training/releases/download/v0.1.0/model-v0.1.0.pkl)

## AI Disclaimer
This documented was refined using ChatGPT 4o.