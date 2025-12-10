import awkward as ak
import numpy as np

from ..transform import Transform


def _normalize(v, axis=-1, eps=1e-7):
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / (norm + eps)


class BackboneBondAngles(Transform):
    """
    Compute backbone bond-angle features (alpha, beta, gamma) per residue,
    encoded as [cos(theta), sin(theta)] for each angle, analogous to PiFold.

    This is the "Angle_features" part of `_dihedrals` in the provided
    PiFold featuriser, factored out as a standalone transform.
    """

    def __init__(self, eps: float = 1e-7):
        self.eps = eps

    def transform_batch(self, batch):
        # Filter residues with missing backbone atoms (N, CA, C)
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

        # Stack backbone as (N, CA, C) and follow the PiFold-style construction
        X = np.stack([N, CA, C], axis=1).to_numpy().reshape(-1, 3)

        # Successive bond directions along the backbone chain
        dX = X[1:] - X[:-1]
        U = _normalize(dX, axis=-1)
        u_0 = U[:-2]
        u_1 = U[1:-1]

        cosD = np.sum(u_0 * u_1, axis=-1)
        cosD = np.clip(cosD, -1 + self.eps, 1 - self.eps)
        D = np.arccos(cosD)

        # Pad so each residue has three angles (alpha_i, gamma_i, beta_{i+1})
        D = np.pad(D, (1, 2), mode="constant", constant_values=0.0)
        D = np.reshape(D, (-1, 3))

        # Zero-out angles at molecule boundaries
        if "residue" in batch.toc:
            boundaries = np.cumsum(batch.toc["residue"].ravel()) - 1
            D[boundaries] = 0.0

        # Encode as [cos(theta), sin(theta)]
        angle_features = np.concatenate([np.cos(D), np.sin(D)], axis=1)
        batch.residue_bond_angles = angle_features
        return batch


