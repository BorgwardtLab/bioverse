import awkward as ak
import numpy as np
from sklearn.neighbors import radius_neighbors_graph

from ..transform import Transform


class EpsGraph(Transform):

    def __init__(self, eps=5.0, mode="connectivity", resolution="atom"):
        super().__init__()
        self.eps = eps
        self.mode = mode
        self.resolution = resolution

    def transform_batch(self, batch):
        edges = []
        values = []
        for coords in batch.molecules.__getattr__(self.resolution + "_pos"):  # type: ignore
            sparse_adj = radius_neighbors_graph(coords, self.eps, mode=self.mode)  # type: ignore
            sparse_adj = sparse_adj.tocoo()  # type: ignore
            edge_index = np.stack([sparse_adj.row, sparse_adj.col], axis=1)
            edges.append(edge_index)
            values.append(sparse_adj.data)
        batch.molecules.molecule_edges = ak.Array(edges)
        if self.mode == "distance":
            batch.molecules.molecule_edge_values = ak.Array(values)
        return batch
