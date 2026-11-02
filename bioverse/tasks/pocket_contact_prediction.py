"""Per-residue peptide-contact prediction on a receptor-only graph."""

from __future__ import annotations

import awkward as ak
import numpy as np

from ..task import Task
from ..transforms.annotate_pocket_contacts import AnnotatePocketContacts
from ..transforms.drop_peptide_residues import DropPeptideResidues
from ..transforms.residue_positions import ResiduePositions


class PocketContactPredictionTask(Task):
    """Predict which protein residues contact a bound peptide.

    Labels come from Cα distance to the peptide (``AnnotatePocketContacts``)
    or are computed here. The peptide is dropped from ``X`` so the encoder
    matches MEROPS (receptor only). ``y["target"]`` is a 0/1 contact per
    remaining residue; ``y["distance"]`` is the nearest peptide Cα distance.
    """

    def __init__(self, radius=8.0, peptide_type="peptide", drop_peptide=True):
        self.radius = float(radius)
        self.peptide_type = peptide_type
        self.drop_peptide = bool(drop_peptide)

    def __call__(self, vbatch, assets, index):
        X = vbatch[index["scene"], index["frame"], index["molecule"]]
        X.resolution = "residue"
        if "residue_pos" not in X.data:
            X = ResiduePositions(mode="CA").transform_batch(X)
        if "residue_contact" not in X.data:
            X = AnnotatePocketContacts(
                radius=self.radius, peptide_type=self.peptide_type
            ).transform_batch(X)
        if self.drop_peptide:
            X = DropPeptideResidues(peptide_type=self.peptide_type).transform_batch(X)
        contacts = np.asarray(
            [int(bool(x)) for x in ak.to_list(ak.ravel(X.residue_contact))],
            dtype=np.float64,
        )
        if "residue_pocket_distance" in X.data:
            distances = np.asarray(
                ak.to_list(ak.ravel(X.residue_pocket_distance)), dtype=np.float64
            )
        else:
            distances = np.full(contacts.shape[0], np.inf, dtype=np.float64)
        counts = X.toc["residue"]
        sizes = (
            np.asarray(ak.to_numpy(ak.ravel(ak.sum(counts, axis=-1))), dtype=np.int64)
            if getattr(counts, "ndim", 0)
            else np.asarray([int(counts)], dtype=np.int64)
        )
        if int(sizes.sum()) != contacts.shape[0]:
            sizes = np.asarray([contacts.shape[0]], dtype=np.int64)
        y = ak.Array(
            {
                "target": ak.unflatten(contacts, sizes),
                "distance": ak.unflatten(distances, sizes),
                "sizes": sizes,
            }
        )
        return X, y
