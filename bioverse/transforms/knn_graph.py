import awkward as ak
import numpy as np
from scipy.spatial import cKDTree

from ..transform import Transform


def _knn_graph(x, k, node_nums):
    # compute knn graph
    batch = np.concatenate([np.ones(n) * v for v, n in enumerate(node_nums)])
    x = (x - np.min(x)) / np.ptp(x)
    x = np.concatenate([x, 2 * x.shape[1] * batch.reshape(-1, 1)], axis=-1)
    dist, col = cKDTree(x).query(x, k=k, distance_upper_bound=x.shape[1])
    row = np.tile(np.arange(col.shape[0]).reshape(-1, 1), (1, k))
    mask = np.isfinite(dist).reshape(-1)
    row, col = row.reshape(-1)[mask], col.reshape(-1)[mask]
    # compute number of edges per graph in batch
    edge2batch = np.concatenate([batch[row], [batch[-1] + 1]])
    sizes = np.argwhere(edge2batch[1:] - edge2batch[:-1]).ravel() + 1
    sizes = np.concatenate([[sizes[0]], np.diff(sizes)])
    edge_index = np.stack([row, col], axis=1)
    edge_index = ak.unflatten(edge_index, sizes)
    # map node indices back to batch
    offset = np.concatenate([[0], np.cumsum(node_nums[:-1])])
    edge_index = edge_index - offset
    return edge_index


class KnnGraph(Transform):

    def __init__(self, k=5, symmetric=True, mode="connectivity", resolution="atom"):
        super().__init__()
        self.k = k
        self.symmetric = symmetric and mode == "connectivity"
        self.mode = mode
        self.resolution = resolution

    def transform_batch(self, batch):
        coords = batch.__getattr__(self.resolution + "_pos").to_numpy()
        node_nums = batch.toc[self.resolution].ravel()
        k = min(len(coords) - 1, self.k)
        batch.molecules.molecule_edges = _knn_graph(coords, k, node_nums)
        return batch

    def _transform_batch(self, batch):
        edges = []
        values = []
        for coords in batch.molecules.__getattr__(self.resolution + "_pos"):  # type: ignore
            # sparse_adj = kneighbors_graph(
            #     coords, min(len(coords) - 1, self.k), mode=self.mode  # type: ignore
            # )
            coords = (
                torch.from_numpy(coords.to_numpy()).float().to(torch.device("cuda"))
            )
            row, col = knn_graph(coords, min(len(coords) - 1, self.k)).cpu().numpy()
            # if self.symmetric:
            #     sparse_adj = sparse_adj + sparse_adj.T.multiply(  # type: ignore
            #         sparse_adj.T > sparse_adj  # type: ignore
            #     )
            # sparse_adj = sparse_adj.tocoo()  # type: ignore
            # edge_index = np.stack([sparse_adj.row, sparse_adj.col], axis=1)
            edge_index = np.stack([row, col], axis=1)
            edges.append(edge_index)
            # values.append(sparse_adj.data)

        if self.mode == "distance":
            batch.molecules.molecule_edge_values = ak.Array(values)
        return batch
