import awkward as ak
import numpy as np
import pytest

from bioverse.factory import BenchmarkFactory, DatasetFactory
from bioverse.utilities import config


@pytest.fixture(scope="module")
def ecacsa_dataset():
    config.workers = 1
    return DatasetFactory("D_ECACSA")


@pytest.fixture(scope="module")
def eccsap_benchmark(ecacsa_dataset):
    return BenchmarkFactory("B_ECCSAP")


@pytest.fixture(scope="module")
def eccsaa_benchmark(ecacsa_dataset):
    return BenchmarkFactory("B_ECCSAA")


def test_dataset_split_counts(ecacsa_dataset):
    split = ecacsa_dataset.split
    assert split["train"].sum() > 0
    assert split["val"].sum() == 1000
    assert split["test"].sum() > 0


def test_eccsap_loader_returns_ec_labels(eccsap_benchmark):
    loader = eccsap_benchmark.loader(
        partition="test",
        batch_size=1,
        batch_on="molecules",
        shuffle=False,
        progress=False,
    )
    (X, y), _ = next(iter(loader))
    target = ak.to_numpy(ak.ravel(y["target"]))[0]
    assert isinstance(target, (str, np.str_))
    assert X.molecules.molecule_ec1 is not None


def test_eccsaa_loader_returns_catalytic_labels(eccsaa_benchmark):
    loader = eccsaa_benchmark.loader(
        partition="test",
        batch_size=1,
        batch_on="molecules",
        shuffle=False,
        progress=False,
    )
    (X, y), _ = next(iter(loader))
    targets = ak.to_numpy(ak.ravel(y["target"])).astype(float)
    assert set(targets) <= {0.0, 1.0}
    assert targets.sum() > 0
