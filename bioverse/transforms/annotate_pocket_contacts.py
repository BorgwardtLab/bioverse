"""Label protein residues that contact a peptide and their Cα distances."""

from __future__ import annotations

import awkward as ak
import numpy as np

from ..transform import Transform
from ..utilities.peptide import flatten_flags, peptide_flags, pocket_distances, pocket_mask


def _flat_numbers(tree):
    out = []

    def walk(node):
        if isinstance(node, (list, tuple)):
            for child in node:
                walk(child)
        else:
            out.append(node)

    walk(tree)
    return out


class AnnotatePocketContacts(Transform):
    """Write ``residue_contact`` and ``residue_pocket_distance`` from peptide Cαs.

    Contacts are protein residues within ``radius`` Å of any peptide Cα.
    Peptide residues are labeled 0 / inf. Apply before dropping the peptide
    so the encoder never sees ligand coordinates.
    """

    def __init__(self, radius=8.0, peptide_type="peptide"):
        self.radius = float(radius)
        self.peptide_type = peptide_type

    def transform_batch(self, batch):
        if "residue_pos" not in batch.data:
            from .residue_positions import ResiduePositions

            batch = ResiduePositions(mode="CA").transform_batch(batch)
        labels = ak.to_list(batch.data["residue_label"])
        types = ak.to_list(batch.data["chain_type"]) if "chain_type" in batch.data else None
        positions = ak.to_list(batch.data["residue_pos"])
        contacts = pocket_mask(
            labels, types, positions, self.radius, peptide_type=self.peptide_type
        )
        distances = pocket_distances(
            labels, types, positions, peptide_type=self.peptide_type
        )
        n_res = int(ak.num(ak.ravel(batch.residue_label), axis=0))
        flat_c = np.asarray(flatten_flags(contacts), dtype=np.int64)
        flat_d = np.asarray(_flat_numbers(distances), dtype=np.float64)
        if flat_c.size != n_res or flat_d.size != n_res:
            pep = np.asarray(
                flatten_flags(peptide_flags(labels, types, self.peptide_type)),
                dtype=bool,
            )
            pos = np.asarray(ak.to_list(ak.ravel(batch.residue_pos)), dtype=np.float64)
            pos = pos.reshape(-1, 3)
            if pep.size != n_res or pos.shape[0] != n_res:
                raise ValueError("pocket contact labels do not align with residues")
            nearest = np.full(n_res, np.inf, dtype=np.float64)
            if pep.any() and (not pep.all()):
                nearest[~pep] = np.linalg.norm(
                    pos[~pep, None, :] - pos[None, pep, :], axis=-1
                ).min(axis=1)
            flat_c = ((~pep) & np.isfinite(nearest) & (nearest <= self.radius)).astype(
                np.int64
            )
            flat_d = nearest
        batch.residues.residue_contact = ak.Array(flat_c)
        batch.residues.residue_pocket_distance = ak.Array(flat_d)
        return batch
