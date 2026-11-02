import awkward as ak
import numpy as np

from ..transform import Transform


def _ca_or_centroid(labels, coords):
    labels = np.asarray(labels)
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)
    if coords.size == 0:
        return np.zeros(3, dtype=np.float64)
    hits = np.flatnonzero(labels == "CA")
    if len(hits):
        return coords[hits[0]]
    return coords.mean(axis=0)


class ResiduePositions(Transform):
    """Compute residue positions from atom coordinates.

    ``CA`` uses the Cα atom when present and otherwise the atom centroid, so
    residues without backbone (ligands, missing CA) stay in the batch.
    """

    def __init__(self, mode="CA"):
        assert mode in ["CA", "COW", "C1'"]
        self.mode = mode

    def transform_batch(self, batch):
        if self.mode == "CA":
            batch.residue_pos = ak.Array(
                [
                    _ca_or_centroid(labels, coords)
                    for labels, coords in zip(
                        ak.to_list(batch.residues.atom_label),
                        ak.to_list(batch.residues.atom_pos),
                    )
                ]
            )
        elif self.mode == "C1'":
            print(list(np.unique(ak.ravel(batch.atom_label))))
            print(
                ak.num(
                    batch.atom_pos[
                        np.logical_or(
                            batch.atom_label == "C1'", batch.atom_label == "C1"
                        )
                    ],
                    axis=0,
                )
            )
            print(ak.sum(batch.toc["atom"]))
            exit()
            batch.residue_pos = batch.atom_pos[batch.atom_label == "C1'"]
        elif self.mode == "COW":
            batch.residue_pos = batch.residues.atom_pos.mean(axis=1)
        return batch
