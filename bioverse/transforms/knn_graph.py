import awkward as ak
import numpy as np
from scipy.spatial import cKDTree

from ..transform import Transform


def _knn_graph(x, k, node_nums):
    node_nums = np.asarray(node_nums, dtype=np.int64).reshape(-1)
    if node_nums.size == 0:
        return ak.Array([])
    if np.any(node_nums <= 0):
        dense = node_nums[node_nums > 0]
        edges = _knn_graph(x, k, dense) if dense.size else ak.Array([])
        edge_list = ak.to_list(edges)
        out = []
        j = 0
        for n in node_nums:
            if n <= 0:
                out.append(np.zeros((0, 2), dtype=np.int64))
            else:
                out.append(edge_list[j])
                j += 1
        return ak.Array(out)
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


def _group_chain_edges(chain_edges, chain_node_nums, chains_per_molecule):
    """Lift per-chain local edges to molecule-local node indices."""
    chain_edges = ak.to_list(chain_edges)
    mol_edges = []
    i = 0
    for n_chains in chains_per_molecule:
        parts = []
        offset = 0
        for _ in range(int(n_chains)):
            edges = np.asarray(chain_edges[i], dtype=np.int64)
            if edges.size:
                parts.append(edges.reshape(-1, 2) + offset)
            offset += int(chain_node_nums[i])
            i += 1
        if parts:
            mol_edges.append(np.concatenate(parts, axis=0))
        else:
            mol_edges.append(np.zeros((0, 2), dtype=np.int64))
    return ak.Array(mol_edges)


class KnnGraph(Transform):

    """Build a k-NN graph over nodes in each molecule.

    Edges are stored on the molecule. By default every chain in a molecule
    shares one graph; set ``link_chains=False`` to build a separate k-NN
    graph per chain (still indexed in molecule-local node order).
    """

    def __init__(
        self,
        k=5,
        symmetric=True,
        mode="connectivity",
        resolution="atom",
        link_chains=True,
    ):
        super().__init__()
        self.k = k
        self.symmetric = symmetric and mode == "connectivity"
        self.mode = mode
        self.resolution = resolution
        self.link_chains = link_chains

    def transform_batch(self, batch):
        coords = batch.__getattr__(self.resolution + "_pos").to_numpy()
        counts = batch.toc[self.resolution]
        if self.resolution == "atom":
            chain_nums = ak.to_numpy(counts.sum(axis=-1).ravel())
            mol_nums = ak.to_numpy(counts.sum(axis=-1).sum(axis=-1).ravel())
        elif self.resolution == "residue":
            chain_nums = ak.to_numpy(counts.ravel())
            mol_nums = ak.to_numpy(counts.sum(axis=-1).ravel())
        else:
            raise ValueError(
                f"resolution must be 'atom' or 'residue', got {self.resolution!r}"
            )
        k = min(len(coords) - 1, self.k)
        if self.link_chains:
            edges = _knn_graph(coords, k, mol_nums)
        else:
            edges = _group_chain_edges(
                _knn_graph(coords, k, chain_nums),
                chain_nums,
                ak.to_numpy(ak.ravel(batch.toc["chain"])),
            )
        batch.molecules.molecule_edges = edges
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
