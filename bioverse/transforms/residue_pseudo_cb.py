import awkward as ak
import numpy as np

from ..transform import Transform


class ResiduePseudoCb(Transform):
    """
    Compute pseudo Cβ coordinates from backbone atoms (N, CA, C), as in
    `_compute_cb` from the PiFold featuriser.

    The resulting coordinates are stored as `batch.residues.residue_cb`.
    """

    def __init__(self):
        super().__init__()

    def transform_batch(self, batch):
        # Filter residues with required backbone atoms
        mask = (
            ak.any(batch.residues.atom_label == "N", axis=1)
            & ak.any(batch.residues.atom_label == "CA", axis=1)
            & ak.any(batch.residues.atom_label == "C", axis=1)
        )
        batch.residues = batch.residues[mask]

        N = batch.atom_pos[batch.atom_label == "N"]
        CA = batch.atom_pos[batch.atom_label == "CA"]
        C = batch.atom_pos[batch.atom_label == "C"]
        assert len(N) == len(CA) == len(C), "Some residues are missing atoms."

        # PiFold-style pseudo Cβ construction
        b = CA - N
        c = C - CA
        a = np.cross(b, c, axis=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA

        batch.residues.residue_cb = Cb
        return batch
