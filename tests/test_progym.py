import awkward as ak
import numpy as np
import pytest

from bioverse.collaters import LongCollater
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


def _assert_batched_gym_labels(X, y):
    labels = ak.to_list(X.molecules.residue_label)
    assert len(labels) == 32
    assert len(y["target"]) == 32
    lengths = {len(seq) for seq in labels}
    assert len(lengths) == 1
    positions = [int(ak.to_list(pos)[0]) for pos in y["position"]]
    assert all(0 <= pos < next(iter(lengths)) for pos in positions)
    assert len({"".join(seq) for seq in labels}) > 1


def test_task_applies_batched_single_chain_mutations(benchmark):
    loader = benchmark.loader(
        partition="test",
        batch_size=32,
        batch_on="mutations",
        shuffle=False,
        progress=False,
    )
    (X, y), _ = next(iter(loader))
    _assert_batched_gym_labels(X, y)


def test_task_applies_batched_mutations_with_workers(benchmark):
    config.workers = 5
    try:
        loader = benchmark.loader(
            partition="test",
            collater=LongCollater(),
            batch_size=32,
            batch_on="mutations",
            shuffle=False,
            progress=False,
        )
        (_, y), data = next(iter(loader))
        assert len(np.asarray(data.y).reshape(-1)) == 32
        assert getattr(data, "mutation_pos", None) is not None
        assert len(np.asarray(data.mutation_pos).reshape(-1)) == 32
    finally:
        config.workers = 1


def _gym_feature_benchmark():
    from bioverse.factory import TransformFactory

    bench = BenchmarkFactory("B_PROGYM_BLAT_ECOLX")
    bench.apply(
        TransformFactory(
            [
                {"ResiduePositions": None},
                {"TokenizeResidues": None},
                {"OneHotResidueFeatures": None},
                {"KnnGraph": {"k": 8, "resolution": "residue"}},
                {"LocalReferenceFrames": {"resolution": "residue"}},
            ]
        )
    )
    return bench


def test_task_featurizes_mutated_tokens():
    config.workers = 1
    benchmark = _gym_feature_benchmark()
    loader = benchmark.loader(
        partition="test",
        collater=LongCollater(),
        batch_size=32,
        batch_on="mutations",
        shuffle=False,
        progress=False,
    )
    (X, y), data = next(iter(loader))
    labels = ak.to_list(X.molecules.residue_label)
    n_res = len(labels[0])
    positions = [int(ak.to_list(pos)[0]) for pos in y["position"]]
    tokens = np.asarray(ak.to_numpy(data.residue_token)).reshape(len(labels), n_res)
    features = np.asarray(ak.to_numpy(data.residue_features)).reshape(
        len(labels), n_res, -1
    )
    mutated_tokens = {int(tokens[i, positions[i]]) for i in range(len(labels))}
    assert len(mutated_tokens) > 1
    assert all(
        int(features[i, positions[i]].argmax()) == int(tokens[i, positions[i]])
        for i in range(len(labels))
    )
    edges = np.asarray(data.molecule_edges)
    assert edges.ndim == 2 and edges.shape[1] == 2


def test_task_collates_long_graphs_with_workers():
    config.workers = 5
    try:
        benchmark = _gym_feature_benchmark()
        loader = benchmark.loader(
            partition="test",
            collater=LongCollater(),
            batch_size=32,
            batch_on="mutations",
            shuffle=False,
            progress=False,
        )
        (_, y), data = next(iter(loader))
        edges = np.asarray(data.molecule_edges)
        feats = np.asarray(data.residue_features)
        assert edges.ndim == 2 and edges.shape[1] == 2
        assert feats.ndim == 2
        assert len(np.asarray(data.y).reshape(-1)) == 32
    finally:
        config.workers = 1
