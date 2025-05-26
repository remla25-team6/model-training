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
from pathlib import Path

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
    # allow up to 0.08 difference
    assert acc == pytest.approx(baseline_accuracy, abs=0.08), (
        f"Seed {seed}: accuracy {acc} differs from baseline {baseline_accuracy}"
    )
    
# Roll-back speed & accuracy test with dvc repro
MODEL   = Path("model") / "model.pkl"
X_TEST  = Path("data")  / "X_test.pkl"
Y_TEST  = Path("data")  / "y_test.pkl"

def _measure_accuracy_at(repo_path, rev):
    subprocess.run(["git", "checkout", rev], cwd=repo_path, check=True)
    subprocess.run(["dvc", "repro", "-q"],   cwd=repo_path, check=True)
    mdl = load(Path(repo_path) / MODEL)
    X   = load(Path(repo_path) / X_TEST)
    y   = load(Path(repo_path) / Y_TEST)
    acc = accuracy_score(y, mdl.predict(X))
    subprocess.run(["dvc", "gc", "-w", "-f", "-q"], cwd=repo_path, check=True)
    subprocess.run(["git", "restore", "."], cwd=repo_path, check=True)
    return acc

@pytest.fixture(scope="session")
def repo_clone(tmp_path_factory):
    root = (
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        )
        .strip()
    )
    clone = tmp_path_factory.mktemp("repo_clone")
    subprocess.run(["git", "clone", "--quiet", root, str(clone)], check=True)
    return clone

@pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("dvc") is None,
    reason="Git and/or DVC not available",
)
def test_repo_rollback_speed_and_accuracy(repo_clone):
    repo = repo_clone

    # measure HEAD accuracy
    head_acc = _measure_accuracy_at(repo, "HEAD")

    # time & measure rollback
    t0 = time.time()
    rollback_acc = _measure_accuracy_at(repo, "HEAD~1")
    elapsed = time.time() - t0

    # speed assertion
    assert elapsed < 30, f"rollback too slow: {elapsed:.2f}s"

    # check if accuracy approx the same
    assert rollback_acc == pytest.approx(head_acc, abs=0.08), (
        f"accuracy drifted: HEAD {head_acc:.3f} vs rollback {rollback_acc:.3f}"
    )