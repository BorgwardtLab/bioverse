import awkward as ak
import numpy as np

from ..transform import Transform


class LinearResidueGraph(Transform):

    def transform_batch(self, batch):
        edges = []
        for coords in batch.molecules.residue_pos:  # type: ignore
            n = len(coords)
            edge_index = [
                np.arange(0, n - 1).tolist() + np.arange(1, n).tolist(),
                np.arange(1, n).tolist() + np.arange(0, n - 1).tolist(),
            ]
            edges.append(edge_index)
        batch.molecules.molecule_graph = ak.Array(edges)
        return batch
