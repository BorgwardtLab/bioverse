import awkward as ak
import numpy as np

from bioverse.data import Batch
from bioverse.tasks.masked_pocket_prediction import MaskedPocketPredictionTask
from bioverse.transforms.annotate_peptide_residues import AnnotatePeptideResidues
from bioverse.transforms.hide_peptide_residues import HidePeptideResidues
from bioverse.transforms.tokenize_residues import TokenizeResidues
from bioverse.utilities import MASKING_TOKEN, PROTEIN_ALPHABET
from bioverse.utilities.peptide import MASK_INDEX, pocket_mask
from tests.dummy.dummy import dummy_scene


def _complex_scene():
    scene = dummy_scene()
    scene["chain_label"] = ak.Array([[["A", "B"]]])
    scene["chain_type"] = ak.Array([[["protein", "peptide"]]])
    scene["residue_label"] = ak.Array([[[["A", "R"], ["G", "K"]]]])
    scene["residue_pos"] = ak.Array(
        [[[[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], [[4.0, 4.0, 0.0], [8.0, 4.0, 0.0]]]]]
    )
    scene["atom_label"] = ak.Array([[[[[["CA"], ["CA"]], [["CA"], ["CA"]]]]]])
    scene["atom_pos"] = ak.Array(
        [[[[[[[0.0, 0.0, 0.0]], [[4.0, 0.0, 0.0]]], [[[4.0, 4.0, 0.0]], [[8.0, 4.0, 0.0]]]]]]]
    )
    return scene


def _complex_batch():
    return Batch(ak.Array([_complex_scene()]))


class _IndexBatch:
    def __init__(self, batch):
        self.batch = batch

    def __getitem__(self, index):
        return self.batch


def test_pocket_mask_protein_near_peptide():
    labels = [["A", "R"], ["G", "K"]]
    types = ["protein", "peptide"]
    pos = [[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], [[4.0, 4.0, 0.0], [8.0, 4.0, 0.0]]]
    mask = pocket_mask(labels, types, pos, radius=8.0)
    assert mask == [[True, True], [False, False]]
    far = pocket_mask(labels, types, pos, radius=4.5)
    assert far == [[False, True], [False, False]]


def test_annotate_and_hide_peptide_residues():
    batch = TokenizeResidues().transform_batch(_complex_batch())
    batch = AnnotatePeptideResidues().transform_batch(batch)
    flags = np.asarray(ak.to_list(ak.ravel(batch.residue_is_peptide)), dtype=bool)
    assert flags.tolist() == [False, False, True, True]
    hidden = HidePeptideResidues().transform_batch(batch)
    labels = list(hidden.residue_label)
    tokens = [int(token) for token in hidden.residue_token]
    assert labels == ["A", "R", MASKING_TOKEN, MASKING_TOKEN]
    assert tokens[:2] == [PROTEIN_ALPHABET.index("A"), PROTEIN_ALPHABET.index("R")]
    assert tokens[2:] == [MASK_INDEX, MASK_INDEX]


def test_masked_pocket_task_targets_protein_only():
    batch = TokenizeResidues().transform_batch(_complex_batch())
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


def test_masked_pocket_task_hides_peptide_after_targets():
    batch = TokenizeResidues().transform_batch(_complex_batch())
    task = MaskedPocketPredictionTask(radius=8.0, hide_peptide=True)
    index = {
        "scene": ak.Array([0]),
        "frame": ak.Array([[0]]),
        "molecule": ak.Array([[[0]]]),
    }
    X, y = task(_IndexBatch(batch), assets={}, index=index)
    targets = np.asarray(ak.to_numpy(ak.ravel(y["target"])))
    assert targets.tolist() == [PROTEIN_ALPHABET.index("A"), PROTEIN_ALPHABET.index("R")]
    labels = list(X.residue_label)
    assert labels[2:] == [MASKING_TOKEN, MASKING_TOKEN]
    tokens = [int(token) for token in X.residue_token]
    assert tokens[2:] == [MASK_INDEX, MASK_INDEX]
