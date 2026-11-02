import awkward as ak
import numpy as np

from ..transform import Transform


class LinearResidueGraph(Transform):

    """Build a graph representation from linear residue graph."""

    def transform_batch(self, batch):
        graphs = []
        pairs = []
        sizes = []
        for coords in batch.molecules.residue_pos:  # type: ignore
            n = len(coords)
            src = np.concatenate([np.arange(0, max(n - 1, 0)), np.arange(1, n)])
            dst = np.concatenate([np.arange(1, n), np.arange(0, max(n - 1, 0))])
            graphs.append([src.tolist(), dst.tolist()])
            if n > 1:
                pairs.append(np.stack([src, dst], axis=1))
            sizes.append(0 if n < 2 else int(src.shape[0]))
        batch.molecules.molecule_graph = ak.Array(graphs)
        # Same pairs as molecule_graph, stored as (n_edges, 2) like KnnGraph.
        stacked = (
            np.concatenate(pairs, axis=0).astype(np.int64)
            if pairs
            else np.zeros((0, 2), dtype=np.int64)
        )
        batch.molecules.molecule_edges = ak.unflatten(stacked, sizes)
        return batch
