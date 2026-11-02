import awkward as ak
import numpy as np

from bioverse.data import Assets, Batch, Split
from bioverse.processors import PdbProcessor
from bioverse.transform import Compose, flatten_transforms
from bioverse.transforms import (
    DeduplicateAtoms,
    ExplodeLigandAtoms,
    FilterSequenceLength,
    Identity,
    KnnGraph,
    LinearResidueGraph,
    NormalizeVector,
    OneHotResidueFeatures,
    ResiduePositions,
    SceneSplit,
    TokenizeByChainType,
    TokenizeResidues,
)
from bioverse.utilities import batched, config

config.workers = 1

from tests.dummy.dummy import dummy_batches, dummy_scene
from bioverse.utilities import (
    BIOCHEMICAL_ATOM_ALPHABET,
    PROTEIN_ALPHABET,
    UNK_TOKEN,
    element_from_atom_name,
)


def _protein_ligand_scene():
    scene = dummy_scene()
    scene["residue_label"] = ak.Array([[[["A", "R", "N", "FMN"]]]])
    scene["atom_label"] = ak.Array(
        [[[[["CA", "CB", "N"], ["CA", "CB", "N"], ["CA", "CB", "N"], ["C1", "C2", "O1"]]]]]
    )
    scene["atom_pos"] = ak.Array(
        [
            [
                [
                    [
                        [
                            [0.2, 0.3, -0.9],
                            [0.2, 0.9, 0.1],
                            [0.7, 0.5, 0.2],
                        ],
                        [
                            [0.4, 0.5, 0.4],
                            [0.6, 0.7, 0.3],
                            [-0.2, 0.1, 0.1],
                        ],
                        [
                            [0.5, -0.6, 0.2],
                            [0.1, -0.7, 0.0],
                            [0.3, 0.3, 0.4],
                        ],
                        [
                            [1.0, 2.0, 3.0],
                            [3.0, 2.0, 1.0],
                            [2.0, 4.0, 2.0],
                        ],
                    ]
                ]
            ]
        ]
    )
    return scene


def test_filter_sequence_length():
    transform = FilterSequenceLength(max_length=1024)
    transformed, split, assets = transform(dummy_batches(), Split(), Assets({}))
    next(transformed)


def test_filter_scenes_updates_split():
    transform = FilterSequenceLength(max_length=1024)
    split = Split(
        {"scene_split": ["train", "train", "val", "test"] * 3},
        default="scene_split",
    )
    transformed, split, assets = transform(dummy_batches(n=12), split, Assets({}))
    list(transformed)
    assert len(split.data["scene_split"]) <= 12


def test_identity():
    transform = Identity()
    transformed, split, assets = transform(dummy_batches(), Split(), Assets({}))
    next(transformed)


def test_knn_residue_graph():
    # this transform needs residue positions
    transform = ResiduePositions(mode="CA")
    transformed, split, assets = transform(dummy_batches(), Split(), Assets({}))

    transform = KnnGraph(k=5, resolution="residue")
    transformed, split, assets = transform(transformed, Split(), Assets({}))
    next(transformed)


def _two_chain_scene():
    return {
        "frame_id": ak.Array([0]),
        "molecule_id": ak.Array([["dummy"]]),
        "chain_label": ak.Array([[["A", "B"]]]),
        "residue_label": ak.Array([[[["A", "A"], ["A", "A"]]]]),
        "atom_label": ak.Array(
            [
                [
                    [
                        [["CA"], ["CA"]],
                        [["CA"], ["CA"]],
                    ]
                ]
            ]
        ),
        "atom_pos": ak.Array(
            [
                [
                    [
                        [[[0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]],
                        [[[1.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]]],
                    ]
                ]
            ]
        ),
    }


def _knn_edges(link_chains):
    import numpy as np

    batch = ResiduePositions(mode="CA").transform_batch(
        Batch(ak.Array([_two_chain_scene()]))
    )
    batch = KnnGraph(
        k=2, resolution="residue", link_chains=link_chains
    ).transform_batch(batch)
    return np.asarray(ak.to_list(ak.flatten(batch.molecule_edges, axis=None))).reshape(
        -1, 2
    )


def test_knn_graph_links_chains_by_default():
    edges = _knn_edges(link_chains=True)
    assert any((src < 2) != (dst < 2) for src, dst in edges)


def test_knn_graph_link_chains_false_keeps_edges_within_chain():
    edges = _knn_edges(link_chains=False)
    assert len(edges) > 0
    assert all((src < 2) == (dst < 2) for src, dst in edges)


def test_linear_graph():
    # this transform needs residue positions
    transform = ResiduePositions(mode="CA")
    transformed, split, assets = transform(dummy_batches(), Split(), Assets({}))

    transform = LinearResidueGraph()
    transformed, split, assets = transform(transformed, Split(), Assets({}))
    next(transformed)


def test_one_hot_residue_features():
    transform = OneHotResidueFeatures()
    transformed, split, assets = transform(dummy_batches(), Split(), Assets({}))
    next(transformed)


def test_random_scene_split():
    transform = SceneSplit(test_size=1, val_size=1, seed=0)
    transformed, split, assets = transform(dummy_batches(), Split(), Assets({}))
    next(transformed)
    names = split.names["random_scene_split"]
    assert set(names) == {"train", "test", "val"}
    assert int(split["train"].sum()) == 8
    assert int(split["test"].sum()) == 1
    assert int(split["val"].sum()) == 1


def test_scene_split_is_stratified_and_shuffled():
    import numpy as np
    from bioverse.adapters.picasso_ship import PicassoShipAdapter
    from bioverse.transforms.scene_split import SceneSplit

    batches, _, _ = PicassoShipAdapter.download(
        n_samples_per_class=20, negative_type="scramble", seed=1
    )
    # Sequential assignment would put all val/test in one class; stratification
    # must keep both classes in every partition.
    transform = SceneSplit(train_size=0.8, test_size=0.1, val_size=0.1, seed=0)
    transformed, split, assets = transform(batches, Split(), Assets({}))
    batch = next(iter(transformed))
    labels = np.asarray(ak.to_numpy(ak.ravel(batch.molecule_present)))
    for part in ("train", "val", "test"):
        part_labels = labels[split[part]]
        assert part_labels.size > 0
        assert (part_labels == 1).any() and (part_labels == 0).any()


def test_residue_positions():
    for mode in ["CA", "COW"]:
        transform = ResiduePositions(mode=mode)
        transformed, split, assets = transform(dummy_batches(), Split(), Assets({}))
        print(next(transformed).residue_label)


def test_residue_positions_uses_ca_when_present():
    import numpy as np

    batch = ResiduePositions(mode="CA").transform_batch(Batch(ak.Array([dummy_scene()])))
    pos = np.asarray(ak.to_list(ak.ravel(batch.residue_pos))).reshape(-1, 3)
    np.testing.assert_allclose(
        pos,
        [
            [0.2, 0.3, -0.9],
            [0.4, 0.5, 0.4],
            [0.5, -0.6, 0.2],
        ],
    )


def test_residue_positions_falls_back_to_centroid():
    import numpy as np

    batch = ResiduePositions(mode="CA").transform_batch(
        Batch(ak.Array([_protein_ligand_scene()]))
    )
    labels = ak.to_list(ak.ravel(batch.residue_label))
    assert labels == ["A", "R", "N", "FMN"]
    pos = np.asarray(ak.to_list(ak.ravel(batch.residue_pos))).reshape(-1, 3)
    np.testing.assert_allclose(pos[0], [0.2, 0.3, -0.9])
    np.testing.assert_allclose(pos[3], [2.0, 8.0 / 3.0, 2.0])


def test_residue_positions_optional_residue_field():
    from tests.dummy.dummy import dummy_scene

    present = dummy_scene()
    present["residue_SASA"] = ak.Array([[[[1.0, 2.0, 3.0]]]])
    missing = dummy_scene()
    batch = Batch(ak.Array([present, missing]))
    assert batch.data["residue_SASA"].layout.is_option
    out = ResiduePositions(mode="CA").transform_batch(batch)
    assert "residue_pos" in out.data
    sasa = out.data["residue_SASA"]
    assert ak.is_none(sasa, axis=0)[1]
    assert len(ak.drop_none(sasa)) == 1
    assert len(ak.ravel(ak.drop_none(sasa))) == 3


def test_batch_toc_ignores_optional_residue_fields():
    from tests.dummy.dummy import dummy_scene

    scene_a = dummy_scene()
    scene_a["residue_SASA"] = ak.Array([[[[1.0, 2.0, 3.0]]]])
    scene_b = dummy_scene()
    batch = Batch(ak.Array([scene_a, scene_b]))
    assert int(ak.sum(ak.ravel(batch.toc["residue"]))) == 6

    merged = batch + Batch(ak.Array([dummy_scene()]))
    assert int(ak.sum(ak.ravel(merged.toc["residue"]))) == 9
    assert len(ak.ravel(merged.residue_label)) == 9


def test_batch_concat_indexes_optional_string_fields():
    from tests.dummy.dummy import dummy_scene

    named = dummy_scene()
    named["molecule_enzyme_name"] = ak.Array(["lysozyme"])
    unnamed = dummy_scene()
    merged = Batch(ak.Array([unnamed])) + Batch(ak.Array([named]))
    scene = ak.Array([0, 1])
    frame = ak.Array([[0], [0]])
    molecule = ak.Array([[0], [0]])
    out = merged[scene, frame, molecule]
    names = ak.to_list(out.data["molecule_enzyme_name"])
    assert names[0] is None
    assert names[1] == ["lysozyme"]


def test_tokenize_residues():
    transform = TokenizeResidues()
    transformed, split, assets = transform(dummy_batches(), Split(), Assets({}))
    next(transformed)


def test_tokenize_residues_maps_unknown_to_unk():
    import numpy as np

    transform = TokenizeResidues()
    batch = transform.transform_batch(Batch(ak.Array([_protein_ligand_scene()])))
    tokens = np.asarray(ak.to_list(ak.ravel(batch.residue_token)))
    assert list(tokens) == [
        PROTEIN_ALPHABET.index("A"),
        PROTEIN_ALPHABET.index("R"),
        PROTEIN_ALPHABET.index("N"),
        transform.unk_index,
    ]
    assert transform.unk_index == len(PROTEIN_ALPHABET)
    assert transform.unk == UNK_TOKEN


def test_one_hot_residue_features_includes_unk():
    import numpy as np

    assets = OneHotResidueFeatures().transform_assets(Assets({}))
    features = np.asarray(assets["residue_features"])
    assert features.shape == (len(PROTEIN_ALPHABET) + 1, len(PROTEIN_ALPHABET) + 1)
    np.testing.assert_array_equal(features[-1], np.eye(len(PROTEIN_ALPHABET) + 1)[-1])


def test_one_hot_residue_features_writes_batch_onehot():
    import numpy as np

    batch = TokenizeResidues().transform_batch(Batch(ak.Array([dummy_scene()])))
    batch = OneHotResidueFeatures().transform_batch(batch)
    n = len(PROTEIN_ALPHABET) + 1
    n_res = int(ak.num(ak.ravel(batch.residue_token), axis=0))
    onehot = np.asarray(ak.to_list(ak.ravel(batch.residue_onehot)), dtype=np.float32)
    onehot = onehot.reshape(n_res, n)
    assert np.allclose(onehot.sum(axis=1), 1.0)


def test_deduplicate_atoms():
    transform = DeduplicateAtoms()
    transformed, split, assets = transform(dummy_batches(), Split(), Assets({}))
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


def test_element_from_atom_name():
    assert element_from_atom_name("CA") == "C"
    assert element_from_atom_name("C1") == "C"
    assert element_from_atom_name("Fe") == "Fe"
    assert element_from_atom_name("1HG") == "H"


def _heterogeneous_scene():
    return {
        "frame_id": ak.Array([0]),
        "molecule_id": ak.Array([["dummy"]]),
        "chain_label": ak.Array([[["A", "B"]]]),
        "chain_type": ak.Array([[["protein", "ligand"]]]),
        "residue_label": ak.Array([[[["A", "C"], ["FMN"]]]]),
        "residue_contact": ak.Array([[[[1, 0], [0]]]]),
        "atom_name": ak.Array(
            [
                [
                    [
                        [["CA", "N"], ["CA", "N"]],
                        [["C1", "N1", "O1"]],
                    ]
                ]
            ]
        ),
        "atom_label": ak.Array(
            [
                [
                    [
                        [["CA", "N"], ["CA", "N"]],
                        [["C", "N", "O"]],
                    ]
                ]
            ]
        ),
        "atom_pos": ak.Array(
            [
                [
                    [
                        [
                            [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                        ],
                        [
                            [[1.0, 0.0, 0.0], [1.2, 0.0, 0.0], [1.4, 0.0, 0.0]],
                        ],
                    ]
                ]
            ]
        ),
    }


def test_explode_ligand_atoms_one_residue_per_atom():
    batch = ExplodeLigandAtoms().transform_batch(
        Batch(ak.Array([_heterogeneous_scene()]))
    )
    labels = ak.to_list(batch.data["residue_label"])
    while isinstance(labels[0][0], list):
        labels = labels[0]
    assert labels[0] == ["A", "C"]
    assert labels[1] == ["C", "N", "O"]
    atoms = ak.to_list(batch.data["atom_label"])
    while isinstance(atoms[0][0][0], list):
        atoms = atoms[0]
    assert atoms[0] == [["CA", "N"], ["CA", "N"]]
    assert atoms[1] == [["C"], ["N"], ["O"]]
    contacts = ak.to_list(ak.ravel(batch.residue_contact))
    assert contacts == [1, 0, 0, 0, 0]


def test_tokenize_by_chain_type_keeps_c_disambiguated():
    import numpy as np

    tokenizer = TokenizeByChainType()
    assets = tokenizer.transform_assets(Assets({}))
    batch = ExplodeLigandAtoms().transform_batch(
        Batch(ak.Array([_heterogeneous_scene()]))
    )
    batch = tokenizer.transform_batch(batch)
    tokens = np.asarray(ak.to_list(ak.ravel(batch.residue_token)))
    protein_c = PROTEIN_ALPHABET.index("C")
    ligand_c = tokenizer.ligand_offset + BIOCHEMICAL_ATOM_ALPHABET.index("C")
    assert tokens[0] == PROTEIN_ALPHABET.index("A")
    assert tokens[1] == protein_c
    assert tokens[2] == ligand_c
    assert protein_c != ligand_c
    features = np.asarray(assets["residue_features"])
    assert features.shape[0] == tokenizer.n_tokens
    assert features.shape[0] > len(PROTEIN_ALPHABET) + 1


def test_exploded_knn_links_protein_residues_to_ligand_atoms():
    batch = ExplodeLigandAtoms().transform_batch(
        Batch(ak.Array([_heterogeneous_scene()]))
    )
    batch = ResiduePositions(mode="CA").transform_batch(batch)
    batch = KnnGraph(k=2, resolution="residue", link_chains=True).transform_batch(batch)
    import numpy as np

    edges = np.asarray(
        ak.to_list(ak.flatten(batch.molecule_edges, axis=None))
    ).reshape(-1, 2)
    assert any((src < 2) != (dst < 2) for src, dst in edges)


def test_deduplicate_atoms_uses_atom_name_when_present():
    scene = _heterogeneous_scene()
    batch = DeduplicateAtoms().transform_batch(Batch(ak.Array([scene])))
    atoms = ak.to_list(batch.data["atom_label"])
    while isinstance(atoms[0][0][0], list):
        atoms = atoms[0]
    assert atoms[1] == [["C", "N", "O"]]


def test_flatten_transforms_unwraps_nested_compose_for_apply_hash():
    pipeline = (
        ResiduePositions(mode="CA"),
        TokenizeResidues(),
        OneHotResidueFeatures(),
        KnnGraph(k=8, resolution="residue"),
    )
    wrapped = Compose(Compose(*pipeline))
    assert Compose(*flatten_transforms(wrapped)).hash() == Compose(*pipeline).hash()


def test_compose_transform_assets_chains_one_hot():
    assets = Compose(OneHotResidueFeatures()).transform_assets(Assets({}))
    features = np.asarray(assets["residue_features"])
    assert features.shape[0] == features.shape[1]
    np.testing.assert_array_equal(features, np.eye(features.shape[0]))
