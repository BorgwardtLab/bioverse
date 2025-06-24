from bioverse.data import Assets, Split
from bioverse.processors import PdbProcessor
from bioverse.transforms import *
from bioverse.utilities import batched, config

config.workers = 1

from tests.dummy.dummy import dummy_batches


def test_filter_sequence_length():
    transform = FilterSequenceLength(max_length=1024)
    transformed, split, assets = transform(dummy_batches(), Split([]), Assets({}))
    next(transformed)


def test_identity():
    transform = Identity()
    transformed, split, assets = transform(dummy_batches(), Split([]), Assets({}))
    next(transformed)


def test_knn_residue_graph():
    # this transform needs residue positions
    transform = ResiduePositions(mode="CA")
    transformed, split, assets = transform(dummy_batches(), Split([]), Assets({}))

    transform = KnnResidueGraph(k=5)
    transformed, split, assets = transform(transformed, Split([]), Assets({}))
    next(transformed)


def test_linear_graph():
    # this transform needs residue positions
    transform = ResiduePositions(mode="CA")
    transformed, split, assets = transform(dummy_batches(), Split([]), Assets({}))

    transform = LinearResidueGraph()
    transformed, split, assets = transform(transformed, Split([]), Assets({}))
    next(transformed)


def test_one_hot_residue_features():
    transform = OneHotResidueFeatures()
    transformed, split, assets = transform(dummy_batches(), Split([]), Assets({}))
    next(transformed)


def test_random_scene_split():
    transform = RandomSceneSplit()
    transformed, split, assets = transform(dummy_batches(), Split([]), Assets({}))
    next(transformed)


def test_residue_positions():
    for mode in ["CA", "COW"]:
        transform = ResiduePositions(mode=mode)
        transformed, split, assets = transform(dummy_batches(), Split([]), Assets({}))
        print(next(transformed).residue_label)


def test_tokenize_residues():
    transform = TokenizeResidues()
    transformed, split, assets = transform(dummy_batches(), Split([]), Assets({}))
    next(transformed)
