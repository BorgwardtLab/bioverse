import awkward as ak

from bioverse.data import Assets, Split
from bioverse.processors import PdbProcessor
from bioverse.transforms import (
    DeduplicateAtoms,
    FilterSequenceLength,
    Identity,
    KnnGraph,
    LinearResidueGraph,
    NormalizeVector,
    OneHotResidueFeatures,
    ResiduePositions,
    SceneSplit,
    TokenizeResidues,
)
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

    transform = KnnGraph(k=5, resolution="residue")
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
    transform = SceneSplit(test_size=1, val_size=1)
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


def test_deduplicate_atoms():
    transform = DeduplicateAtoms()
    transformed, split, assets = transform(dummy_batches(), Split([]), Assets({}))
    batch = next(transformed)
    for atoms in ak.to_list(batch.residues.atom_label):
        assert len(atoms) == len(set(atoms))


def test_normalize_vector():
    import numpy as np

    class Obj:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def __getattr__(self, name):
            return self.__dict__[name]

        def __setattr__(self, name, value):
            self.__dict__[name] = value

    vectors = ak.Array([[[1.0, 0.0, 0.0], [0.0, 3.0, 4.0]]])
    batch = Obj(atom_force=vectors)
    out = NormalizeVector("atom_force").transform_batch(batch)
    norms = ak.ravel(np.sqrt(ak.sum(out.atom_force * out.atom_force, axis=-1)))
    np.testing.assert_allclose(ak.to_numpy(norms), [1.0, 1.0], rtol=1e-5)
