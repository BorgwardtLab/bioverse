import awkward as ak
import numpy as np

from ..transform import Transform


class Rbf(Transform):

    def __init__(self, D_min=0.0, D_max=20.0, D_count=16):
        super().__init__()
        self.D_min = D_min
        self.D_max = D_max
        self.D_count = D_count

    def transform_batch(self, batch):
        edge_index = batch.molecules.molecule_edges
        pos = batch.molecules.residue_pos
        edge_vectors = pos[edge_index[:, :, 0]] - pos[edge_index[:, :, 1]]
        D = np.sqrt(ak.sum(edge_vectors**2, axis=-1))
        D_mu = np.linspace(self.D_min, self.D_max, self.D_count)
        D_mu = D_mu.reshape(1, 1, -1)
        D_sigma = (self.D_max - self.D_min) / self.D_count
        rbf = np.exp(-(((D - D_mu) / D_sigma) ** 2))
        batch.molecules.molecule_edge_rbf = rbf
        return batch
