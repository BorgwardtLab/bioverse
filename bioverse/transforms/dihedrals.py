import awkward as ak
import numpy as np

from ..transform import Transform
from ..utilities import normalize


class Dihedrals(Transform):

    def __init__(self, eps=1e-7):
        self.eps = eps

    def transform_batch(self, batch):
        # From https://github.com/drorlab/gvp-pytorch.git

        # filter residues with missing atoms
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

        X = np.stack([N, CA, C], axis=1).to_numpy().reshape(-1, 3)
        U = normalize(X[1:] - X[:-1], axis=-1)
        u_2, u_1, u_0 = U[:-2], U[1:-1], U[2:]
        n_2 = normalize(np.cross(u_2, u_1), axis=-1)
        n_1 = normalize(np.cross(u_1, u_0), axis=-1)
        cosD = np.sum(n_2 * n_1, axis=-1)
        cosD = np.clip(cosD, -1 + self.eps, 1 - self.eps)
        D = np.sign(np.sum(u_2 * n_1, axis=-1)) * np.arccos(cosD)
        D = np.pad(D, (1, 2), mode="constant", constant_values=0.0)
        D = np.reshape(D, (-1, 3))
        D[np.cumsum(batch.toc["residue"].ravel()) - 1] = 0.0  # todo: fix this properly
        dihedrals = np.concatenate([np.cos(D), np.sin(D)], axis=1)
        batch.residue_dihedrals = dihedrals
        return batch
