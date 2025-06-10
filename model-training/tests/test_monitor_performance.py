"""
Quick performance-regression checks for inference & training.

Run and save once:   pytest --benchmark-save=perf
Compare:    pytest --benchmark-compare
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Tuple

import psutil
import pytest
from joblib import load
from sklearn.utils import shuffle
from restaurant_sentiment.train import train

# Tunable limits
N_PRED = 2_000  # samples per inference batch
INFER_SLOW_TOL = 0.25  # ≥25 % slower than baseline ⇒ fail
TRAIN_SLOW_TOL = 0.25
MEM_LEAK_LIMIT = 10 * 1024  # allowed MEM increase during predict (10 KB)
TRAIN_MEM_LIMIT = 100 * 1024**2  # allowed extra MEM during training (100 MB)


@pytest.fixture(scope="session")
def data_dir() -> Path:
    root = Path(__file__).resolve().parents[2]  # repo/model-training/
    return Path(os.getenv("MODEL_TRAINING_DATA_DIR", root / "data"))


@pytest.fixture(scope="session")
def trained(tmp_path_factory, data_dir: Path) -> Tuple[object, Path]:
    """Train once per session; return (model, data_path)."""
    out = tmp_path_factory.mktemp("perf_model")
    train(data_path=data_dir, model_path=str(out), random_state=0)
    return load(out / "model.pkl"), data_dir


def current_memory() -> int:
    """Return current RSS in bytes."""
    return psutil.Process().memory_info().rss


# inference latency + memory leak
@pytest.mark.benchmark(group="inference")
def test_inference_latency_and_leak(trained, benchmark):
    model, data_dir = trained
    X = shuffle(load(data_dir / "X_test.pkl"), random_state=0)[
        :N_PRED
    ]  # samples for inference

    benchmark(model.predict, X)
    mean_batch = benchmark.stats.stats.mean  # avg seconds per batch

    # compare to previous saved run (if --benchmark-compare was used)
    baseline = getattr(benchmark.stats, "baseline", None)
    if baseline:
        assert mean_batch < baseline["mean"] * (1 + INFER_SLOW_TOL)

    # mem leak check before/after an extra prediction call
    before = current_memory()
    model.predict(X)
    assert current_memory() - before < MEM_LEAK_LIMIT


# training speed + peak memory
@pytest.mark.benchmark(group="training")
def test_monitor_6_training_speed_and_ram(data_dir, tmp_path, benchmark):
    rss_inc: dict[str, int] = {}

    def _train() -> float:
        startmemory = current_memory()
        t0 = time.perf_counter()
        train(data_path=data_dir, model_path=str(tmp_path / "bench"), random_state=0)
        rss_inc["delta"] = current_memory() - startmemory
        return time.perf_counter() - t0

    benchmark(_train)
    train_mean = benchmark.stats.stats.mean

    baseline = getattr(benchmark.stats, "baseline", None)
    if baseline:
        assert train_mean < baseline["mean"] * (1 + TRAIN_SLOW_TOL)

    assert rss_inc["delta"] < TRAIN_MEM_LIMIT
