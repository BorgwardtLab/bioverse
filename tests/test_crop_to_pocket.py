import awkward as ak
import numpy as np

from bioverse.data import Batch
from bioverse.factory import TransformFactory
from bioverse.tasks.masked_pocket_prediction import MaskedPocketPredictionTask
from bioverse.transforms.crop_to_pocket import CropToPocket
from bioverse.transforms.tokenize_residues import TokenizeResidues
from bioverse.utilities import PROTEIN_ALPHABET
from tests.dummy.dummy import dummy_scene


def _far_protein_scene():
    scene = dummy_scene()
    scene["chain_label"] = ak.Array([[["A", "B"]]])
    scene["chain_type"] = ak.Array([[["protein", "peptide"]]])
    scene["residue_label"] = ak.Array([[[["A", "R", "D"], ["G", "K"]]]])
    scene["residue_number"] = ak.Array([[[[1, 2, 81], [301, 302]]]])
    scene["residue_pos"] = ak.Array(
        [[[[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [100.0, 0.0, 0.0]], [[4.0, 4.0, 0.0], [8.0, 4.0, 0.0]]]]]
    )
    atom_label = ak.to_list(scene["atom_label"])
    atom_label[0][0][0] = [[["CA"], ["CA"], ["CA"]], [["CA"], ["CA"]]]
    scene["atom_label"] = ak.Array(atom_label)
    atom_pos = ak.to_list(scene["atom_pos"])
    atom_pos[0][0][0] = [
        [[[0.0, 0.0, 0.0]], [[4.0, 0.0, 0.0]], [[100.0, 0.0, 0.0]]],
        [[[4.0, 4.0, 0.0]], [[8.0, 4.0, 0.0]]],
    ]
    scene["atom_pos"] = ak.Array(atom_pos)
    return scene


class _IndexBatch:
    def __init__(self, batch):
        self.batch = batch

    def __getitem__(self, index):
        return self.batch


def test_crop_drops_distal_protease_keeps_peptide():
    batch = TokenizeResidues().transform_batch(Batch(ak.Array([_far_protein_scene()])))
    cropped = CropToPocket(radius=8.0).transform_batch(batch)
    labels = [str(x) for x in ak.to_list(ak.ravel(cropped.residue_label))]
    assert "D" not in labels
    assert labels == ["A", "R", "G", "K"]
    pocket = np.asarray(ak.to_list(ak.ravel(cropped.residue_is_pocket)), dtype=bool)
    assert pocket.tolist() == [True, True, False, False]


def test_task_after_crop_masks_all_remaining_protein():
    batch = TokenizeResidues().transform_batch(Batch(ak.Array([_far_protein_scene()])))
    batch = CropToPocket(radius=8.0).transform_batch(batch)
    task = MaskedPocketPredictionTask(radius=8.0, hide_peptide=False)
    index = {
        "scene": ak.Array([0]),
        "frame": ak.Array([[0]]),
        "molecule": ak.Array([[[0]]]),
    }
    X, y = task(_IndexBatch(batch), assets={}, index=index)
    targets = np.asarray(ak.to_numpy(ak.ravel(y["target"])))
    assert targets.tolist() == [PROTEIN_ALPHABET.index("A"), PROTEIN_ALPHABET.index("R")]
    mask = np.asarray(ak.to_list(ak.ravel(X.molecules.residue_mask)), dtype=bool)
    assert mask.tolist() == [True, True, False, False]


def test_transform_factory_builds_crop():
    factory = TransformFactory([{"CropToPocket": {"radius": 8.0}}])
    batch = TokenizeResidues().transform_batch(Batch(ak.Array([_far_protein_scene()])))
    cropped = factory.transform_batch(batch)
    labels = [str(x) for x in ak.to_list(ak.ravel(cropped.residue_label))]
    assert labels == ["A", "R", "G", "K"]


def test_crop_then_knn_keeps_every_molecule():
    from bioverse.transforms.knn_graph import KnnGraph

    scene = _far_protein_scene()
    batch = TokenizeResidues().transform_batch(Batch(ak.Array([scene, scene])))
    cropped = CropToPocket(radius=8.0).transform_batch(batch)
    labels = [str(x) for x in ak.to_list(ak.ravel(cropped.residue_label))]
    assert labels == ["A", "R", "G", "K", "A", "R", "G", "K"]
    knn = KnnGraph(k=2, resolution="residue").transform_batch(cropped)
    edges = ak.to_list(knn.molecule_edges)
    assert len(edges) == 2
    assert all(len(graph) > 0 for graph in edges)


def test_task_on_two_cropped_molecules_has_targets():
    scene = _far_protein_scene()
    batch = TokenizeResidues().transform_batch(Batch(ak.Array([scene, scene])))
    batch = CropToPocket(radius=8.0).transform_batch(batch)
    task = MaskedPocketPredictionTask(radius=8.0, hide_peptide=False)
    index = {
        "scene": ak.Array([0, 1]),
        "frame": ak.Array([[0], [0]]),
        "molecule": ak.Array([[[0]], [[0]]]),
    }
    X, y = task(_IndexBatch(batch), assets={}, index=index)
    targets = np.asarray(ak.to_numpy(ak.ravel(y["target"])))
    assert targets.tolist() == [
        PROTEIN_ALPHABET.index("A"),
        PROTEIN_ALPHABET.index("R"),
        PROTEIN_ALPHABET.index("A"),
        PROTEIN_ALPHABET.index("R"),
    ]
    sizes = np.asarray(ak.to_numpy(ak.ravel(y["sizes"])), dtype=int)
    assert sizes.tolist() == [2, 2]
    pocket = np.asarray(ak.to_list(ak.ravel(X.residue_is_pocket)), dtype=bool)
    assert pocket.tolist() == [True, True, False, False, True, True, False, False]
