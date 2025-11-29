import awkward as ak
import numpy as np

from ..transform import Transform


class RamachandranAngles(Transform):

    def transform_batch(self, batch):
        X = batch.residues.residue_backbone
        N, CA, C = X[:, 0], X[:, 1], X[:, 2]

        def compute_dihedral(p0, p1, p2, p3):
            b0 = p0 - p1
            b1 = p2 - p1
            b2 = p3 - p2
            b1_norm = b1 / np.linalg.norm(b1, axis=1, keepdims=True)
            v = b0 - (np.einsum("ij,ij->i", b0, b1_norm)[:, None]) * b1_norm
            w = b2 - (np.einsum("ij,ij->i", b2, b1_norm)[:, None]) * b1_norm
            x = np.einsum("ij,ij->i", v, w)
            y = np.einsum("ij,ij->i", np.cross(b1_norm, v), w)
            return np.arctan2(y, x)

        phi = np.zeros(len(N), dtype=float)
        psi = np.zeros(len(N), dtype=float)
        if len(N) >= 2:
            phi[1:] = compute_dihedral(C[:-1], N[1:], CA[1:], C[1:])
            psi[:-1] = compute_dihedral(N[:-1], CA[:-1], C[:-1], N[1:])
        batch.residue_phi = phi
        batch.residue_psi = psi
        return batch
