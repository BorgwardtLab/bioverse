import awkward as ak
import numpy as np

from bioverse.data import Batch
from bioverse.tasks.masked_pocket_prediction import MaskedPocketPredictionTask
from bioverse.transforms.annotate_peptide_residues import AnnotatePeptideResidues
from bioverse.transforms.hybrid_pocket_features import FEATURE_WIDTH, HybridPocketFeatures
from bioverse.transforms.peptide_atom_context import PeptideAtomContext
from bioverse.transforms.tokenize_residues import TokenizeResidues
from bioverse.utilities import PROTEIN_ALPHABET


def _peptide_scene():
    return {
        "frame_id": ak.Array([0]),
        "molecule_id": ak.Array([["dummy"]]),
        "chain_label": ak.Array([[["A", "B"]]]),
        "chain_type": ak.Array([[["protein", "peptide"]]]),
        "residue_label": ak.Array([[[["A", "R"], ["F"]]]]),
        "residue_number": ak.Array([[[[1, 2], [3]]]]),
        "residue_pos": ak.Array([[[[ [0.0, 0.0, 0.0], [4.0, 0.0, 0.0] ], [[4.0, 4.0, 0.0]] ]]]),
        "atom_name": ak.Array(
            [
                [
                    [
                        [["N", "CA", "C", "O"], ["N", "CA", "C", "O", "CB"]],
                        [["N", "CA", "C", "O", "CB", "CG"]],
                    ]
                ]
            ]
        ),
        "atom_label": ak.Array(
            [
                [
                    [
                        [["N", "CA", "C", "O"], ["N", "CA", "C", "O", "CB"]],
                        [["N", "CA", "C", "O", "CB", "CG"]],
                    ]
                ]
            ]
        ),
        "atom_pos": ak.Array(
            [
                [
                    [
                        [
                            [
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0],
                                [1.5, 1.0, 0.0],
                            ],
                            [
                                [4.0, 1.0, 0.0],
                                [4.0, 0.0, 0.0],
                                [5.0, 0.0, 0.0],
                                [5.5, 1.0, 0.0],
                                [4.0, -1.5, 0.0],
                            ],
                        ],
                        [
                            [
                                [4.0, 5.0, 0.0],
                                [4.0, 4.0, 0.0],
                                [5.0, 4.0, 0.0],
                                [5.5, 5.0, 0.0],
                                [4.0, 2.5, 0.0],
                                [3.0, 1.5, 0.0],
                            ]
                        ],
                    ]
                ]
            ]
        ),
    }


def _batch():
    batch = Batch(ak.Array([_peptide_scene()]))
    batch.resolution = "residue"
    batch = TokenizeResidues().transform_batch(batch)
    return AnnotatePeptideResidues().transform_batch(batch)


class _IndexBatch:
    def __init__(self, batch):
        self.batch = batch

    def __getitem__(self, index):
        return self.batch


def test_peptide_atom_context_keeps_protein_and_explodes_sidechains():
    batch = PeptideAtomContext(sidechains=True).transform_batch(_batch())
    labels = [str(x) for x in ak.to_list(ak.ravel(batch.residue_label))]
    names = [str(x) for x in ak.to_list(ak.ravel(batch.residue_atom_name))]
    is_atom = np.asarray(ak.to_list(ak.ravel(batch.residue_is_atom)), dtype=bool)
    assert labels[:2] == ["A", "R"]
    assert is_atom.tolist() == [False, False] + [True] * (len(labels) - 2)
    assert "CG" in names
    assert names.count("CA") == 1


def test_peptide_atom_context_uncond_drops_sidechains():
    batch = PeptideAtomContext(sidechains=False).transform_batch(_batch())
    names = [str(x) for x in ak.to_list(ak.ravel(batch.residue_atom_name))]
    assert "CG" not in names and "CB" not in names
    assert set(names[2:]) <= {"N", "CA", "C", "O"}


def test_hybrid_features_and_pocket_task():
    batch = PeptideAtomContext(sidechains=True).transform_batch(_batch())
    batch = HybridPocketFeatures().transform_batch(batch)
    feat = np.asarray(ak.to_numpy(batch.residue_features), dtype=np.float32).reshape(-1, FEATURE_WIDTH)
    assert feat.shape[1] == FEATURE_WIDTH
    assert feat[0, -2] == 1.0
    assert feat[2, -1] == 1.0
    task = MaskedPocketPredictionTask(radius=8.0, hide_peptide=False)
    index = {
        "scene": ak.Array([0]),
        "frame": ak.Array([[0]]),
        "molecule": ak.Array([[[0]]]),
    }
    X, y = task(_IndexBatch(batch), assets={}, index=index)
    targets = np.asarray(ak.to_numpy(ak.ravel(y["target"])))
    assert PROTEIN_ALPHABET.index("A") in targets.tolist() or PROTEIN_ALPHABET.index("R") in targets.tolist()
    mask = np.asarray(ak.to_list(ak.ravel(X.molecules.residue_mask)), dtype=bool)
    assert not mask[2:].any()
