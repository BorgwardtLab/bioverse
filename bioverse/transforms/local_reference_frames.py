import awkward as ak
import numpy as np

from ..transform import Transform


class LocalReferenceFrames(Transform):

    def __init__(self, resolution="atom"):
        self.resolution = resolution

    def transform_batch(self, batch):
        Ts, Rs = [], []
        for pos, edges in zip(
            batch.molecules.__getattr__(self.resolution + "_pos"),
            batch.molecules.molecule_edges,
        ):
            # compute neighborhoods by translating to current position
            a, b = edges[:, 0], edges[:, 1]
            n_pos = np.array(pos[b] - pos[a])  # neighbor positions
            # compute principal components of the local neighborhood for each position
            for i in range(len(pos)):
                n_pos_i = n_pos[a == i]
                # SVD
                U, S, Vt = np.linalg.svd(n_pos_i)
                # compute rotation matrix from principal components
                R = Vt.T
                if np.linalg.det(R) < 0:
                    R[:, 2] *= -1
                Rs.append(R)
                Ts.append(pos[i])
        setattr(batch, self.resolution + "_frame_T", ak.Array(Ts))
        setattr(batch, self.resolution + "_frame_R", ak.Array(Rs))
        return batch
