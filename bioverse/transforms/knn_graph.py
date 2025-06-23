import awkward as ak
import numpy as np
from sklearn.neighbors import kneighbors_graph

from ..transform import Transform


class KnnGraph(Transform):

    def __init__(self, k=5, symmetric=True, mode="connectivity", resolution="atom"):
        super().__init__()
        self.k = k
        self.symmetric = symmetric and mode == "connectivity"
        self.mode = mode
        self.resolution = resolution

    def transform_batch(self, batch):
        edges = []
        values = []
        for coords in batch.molecules.__getattr__(self.resolution + "_pos"):  # type: ignore
            sparse_adj = kneighbors_graph(
                coords, min(len(coords) - 1, self.k), mode=self.mode
            )
            if self.symmetric:
                sparse_adj = sparse_adj + sparse_adj.T.multiply(  # type: ignore
                    sparse_adj.T > sparse_adj  # type: ignore
                )
            sparse_adj = sparse_adj.tocoo()  # type: ignore
            edge_index = np.vstack([sparse_adj.row, sparse_adj.col])
            edges.append(edge_index)
            values.append(sparse_adj.data)
        batch.molecules.molecule_graph = ak.Array(edges)
        if self.mode == "distance":
            batch.molecules.molecule_graph_distances = ak.Array(values)
        return batch
