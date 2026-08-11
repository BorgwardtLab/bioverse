import awkward as ak
import numpy as np

from ..transform import Transform


def _local_reference_frames(pos, edges):
    pos = np.asarray(pos)
    edges = np.asarray(edges)
    V = len(pos)
    R = np.broadcast_to(np.eye(3), (V, 3, 3)).copy()
    if len(edges) == 0:
        return pos, R

    a = edges[:, 0]
    n_pos = pos[edges[:, 1]] - pos[a]
    counts = np.bincount(a, minlength=V)
    order = np.argsort(a)
    n_sorted = n_pos[order]
    starts = np.cumsum(np.concatenate([[0], counts[:-1]]))

    for k in np.unique(counts[counts > 0]):
        verts = np.where(counts == k)[0]
        idx = starts[verts][:, None] + np.arange(k)[None, :]
        _, _, Vt = np.linalg.svd(n_sorted[idx], full_matrices=True)
        Rk = np.transpose(Vt, (0, 2, 1))
        flip = np.linalg.det(Rk) < 0
        Rk[flip, :, 2] *= -1
        R[verts] = Rk

    return pos, R


class LocalReferenceFrames(Transform):

    """Compute or add local reference frames features to batches."""

    def __init__(self, resolution="atom"):
        self.resolution = resolution

    def transform_batch(self, batch):
        pos_field = self.resolution + "_pos"
        pos_list = batch.molecules.__getattr__(pos_field)
        edges_list = batch.molecules.molecule_edges

        all_pos = []
        all_a = []
        all_npos = []
        offsets = [0]
        for pos, edges in zip(pos_list, edges_list):
            pos = np.asarray(ak.flatten(ak.Array(pos), axis=None)).reshape(-1, 3)
            edges = np.asarray(edges)
            if len(edges) > 0:
                all_a.append(edges[:, 0] + offsets[-1])
                all_npos.append(pos[edges[:, 1]] - pos[edges[:, 0]])
            all_pos.append(pos)
            offsets.append(offsets[-1] + len(pos))

        pos_all = np.concatenate(all_pos).reshape(-1, 3)
        V = len(pos_all)
        R_all = np.broadcast_to(np.eye(3), (V, 3, 3)).copy()
        if all_a:
            a = np.concatenate(all_a)
            n_pos = np.concatenate(all_npos)
            counts = np.bincount(a, minlength=V)
            order = np.argsort(a)
            n_sorted = n_pos[order]
            starts = np.cumsum(np.concatenate([[0], counts[:-1]]))

            for k in np.unique(counts[counts > 0]):
                verts = np.where(counts == k)[0]
                idx = starts[verts][:, None] + np.arange(k)[None, :]
                _, _, Vt = np.linalg.svd(n_sorted[idx], full_matrices=True)
                Rk = np.transpose(Vt, (0, 2, 1))
                flip = np.linalg.det(Rk) < 0
                Rk[flip, :, 2] *= -1
                R_all[verts] = Rk

        setattr(batch, self.resolution + "_frame_T", ak.Array(pos_all))
        setattr(batch, self.resolution + "_frame_R", ak.Array(R_all))
        return batch
