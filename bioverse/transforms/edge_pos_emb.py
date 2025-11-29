import awkward as ak
import numpy as np

from ..transform import Transform
from ..utilities import normalize


class EdgePosEmb(Transform):

    def __init__(self, num_embeddings=16, period_range=[2, 1000]):
        self.num_embeddings = num_embeddings
        self.period_range = period_range

    def transform_batch(self, batch):
        edge_index = ak.flatten(batch.molecule_edges, axis=1).to_numpy()
        d = edge_index[:, 0] - edge_index[:, 1]
        frequency = np.exp(
            np.arange(0, self.num_embeddings, 2)
            * -(np.log(10000.0) / self.num_embeddings)
        )
        angles = np.expand_dims(d, -1) * frequency
        pos_emb = np.concatenate((np.cos(angles), np.sin(angles)), -1)
        batch.molecule_edge_pos_emb = ak.unflatten(
            pos_emb, ak.num(batch.molecule_edges, axis=1)
        )
        return batch
