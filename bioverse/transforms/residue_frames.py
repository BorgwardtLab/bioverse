import awkward as ak
import numpy as np

from ..transform import Transform


def normalize(v, axis=-1, eps=1e-8):
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / (norm + eps)


class ResidueFrames(Transform):

    def __init__(self):
        pass

    def transform_batch(self, batch):
        # filter residues with missing atoms
        mask = (
            ak.any(batch.residues.atom_label == "N", axis=1)
            & ak.any(batch.residues.atom_label == "CA", axis=1)
            & ak.any(batch.residues.atom_label == "C", axis=1)
        )
        batch.residues = batch.residues[mask]
        # get N-CA-C-O coords
        N = batch.atom_pos[batch.atom_label == "N"]
        CA = batch.atom_pos[batch.atom_label == "CA"]
        C = batch.atom_pos[batch.atom_label == "C"]
        assert len(N) == len(CA) == len(C), "Some residues are missing atoms."
        # compute reference frames as unit vectors
        z = normalize(C - CA)
        ref_vec = normalize(N - CA)
        y = normalize(np.cross(z, ref_vec))
        x = normalize(np.cross(y, z))
        R = np.stack([x, y, z], axis=-1)
        batch.residues.residue_frame_T = CA
        batch.residues.residue_frame_R = R
        return batch
