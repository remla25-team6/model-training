import os
import pytest
from joblib import load
from sklearn.metrics import accuracy_score
from restaurant_sentiment.train import train
from restaurant_sentiment.get_data import download_dataset
from restaurant_sentiment.preprocess import main
import subprocess
import shutil
import time

@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    data_url = "https://raw.githubusercontent.com/proksch/restaurant-sentiment/main/a1_RestaurantReviews_HistoricDump.tsv"
    # 1. Download
    data_dir = tmp_path_factory.mktemp("data")
    raw_tsv = data_dir / "RestaurantReviews_HistoricDump.tsv"
    download_dataset(data_url, raw_tsv)
    # 2. Preprocess
    main(data_path=str(data_dir), filepath=str(raw_tsv))
    return str(data_dir)

@pytest.fixture(scope="session")
def baseline_accuracy(data_dir, tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("model0")
    # train with the default (seed=0)
    train(data_path=data_dir, model_path=str(model_dir), random_state=0)
    model = load(os.path.join(model_dir, "model.pkl"))
    X_test = load(os.path.join(data_dir, "X_test.pkl"))
    y_test = load(os.path.join(data_dir, "y_test.pkl"))
    return accuracy_score(y_test, model.predict(X_test))

@pytest.mark.parametrize("seed", [1, 2, 3])
def test_accuracy_stability(data_dir, baseline_accuracy, tmp_path, seed):
    model_dir = tmp_path / f"model_seed_{seed}"
    train(data_path=data_dir, model_path=str(model_dir), random_state=seed)
    model = load(os.path.join(model_dir, "model.pkl"))
    X_test = load(os.path.join(data_dir, "X_test.pkl"))
    y_test = load(os.path.join(data_dir, "y_test.pkl"))
    acc = accuracy_score(y_test, model.predict(X_test))
    # allow up to 8% relative difference
    assert acc == pytest.approx(baseline_accuracy, abs=0.08), (
        f"Seed {seed}: accuracy {acc} differs from baseline {baseline_accuracy}"
    )
    
# Roll-back speed & correctness with Git checkout + dvc repro (+ cleanup)
MODEL_REL = os.path.join("model", "model.pkl")
X_REL     = os.path.join("data", "X_test.pkl")
Y_REL     = os.path.join("data", "y_test.pkl")

@pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("dvc") is None,
    reason="Git and/or DVC not available",
)

def _accuracy(repo_path):
    mdl = load(os.path.join(repo_path, MODEL_REL))
    X   = load(os.path.join(repo_path, X_REL))
    y   = load(os.path.join(repo_path, Y_REL))
    return accuracy_score(y, mdl.predict(X))

def _checkout_repro_cleanup(repo, commit):
    subprocess.run(["git", "checkout", commit], cwd=repo, check=True)
    subprocess.run(["dvc", "repro", "-q"], cwd=repo, check=True)
    acc = _accuracy(repo)
    # wipe any un-needed workspace files / tmp cache
    subprocess.run(["dvc", "gc", "-w", "-f", "-q"], cwd=repo, check=True)
    return acc

def test_repo_rollback_speed_and_correctness():
    repo = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
    orig_rev = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    target = os.getenv("ROLLBACK_COMMIT", "HEAD~1")

    # build & evaluate model at HEAD
    acc_head = _checkout_repro_cleanup(repo, orig_rev)

    try:
        t0 = time.time()
        # rollback to older commit, rebuild & measure
        acc_old = _checkout_repro_cleanup(repo, target)
        elapsed = time.time() - t0

        assert acc_head != pytest.approx(acc_old), "model did not change after rollback"
        assert elapsed < 30, f"rollback too slow: {elapsed:.2f}s"
    finally:
        # always restore workspace for the rest of the suite
        _checkout_repro_cleanup(repo, orig_rev)