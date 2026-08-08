import awkward as ak
import numpy as np
import pytest

from bioverse.factory import BenchmarkFactory
from bioverse.utilities import config

ASSAY = "BLAT_ECOLX_Stiffler_2015"


@pytest.fixture(scope="module")
def benchmark():
    config.workers = 1
    return BenchmarkFactory("B_PROGYM_BLAT_ECOLX")


def test_dataset_has_single_assay(benchmark):
    assert len(benchmark.dataset.toc) == 1
    batch = next(iter(benchmark.dataset.shards))
    assay_id = batch.data["molecule_id"][0][0][0]
    assert assay_id == ASSAY
    assert benchmark.dataset.toc["mutations"][0] == 4996


def test_task_returns_mutation_targets(benchmark):
    loader = benchmark.loader(
        partition="test",
        batch_size=1,
        batch_on="mutations",
        shuffle=False,
        progress=False,
    )
    (X, y), _ = next(iter(loader))

    assert "target" in y.fields
    assert "position" in y.fields
    assert len(y["target"]) == 1
    assert isinstance(y["target"][0].item(), (float, np.floating))
    assert len(y["position"][0]) >= 1
    assert y["position"][0][0].item() >= 0

    mutated = X.molecules.residue_label[0]
    assert len(mutated) > 0
