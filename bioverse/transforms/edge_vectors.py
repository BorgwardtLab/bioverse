import awkward as ak
import numpy as np

from ..transform import Transform


class EdgeVectors(Transform):

    def transform_batch(self, batch):
        edge_index = batch.molecules.molecule_edges
        pos = batch.molecules.residue_pos
        edge_vectors = pos[edge_index[:, :, 0]] - pos[edge_index[:, :, 1]]
        norm = np.sqrt(ak.sum(edge_vectors**2, axis=-1))
        batch.molecules.molecule_edge_vectors = edge_vectors / (norm + 1e-8)
        return batch
