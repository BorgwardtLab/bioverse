import awkward as ak
import numpy as np
import torch
from torch_geometric.data import Data

from ..framework import Framework


class PygData(Data):

    def uncollate(self, y):
        if "sizes" in self:
            y = ak.unflatten(y, self.sizes, axis=0)
        y = ak.Array({"target": y})
        return y


class PygFramework(Framework):

    @classmethod
    def collate(cls, X, y=None, attr=[]):
        if X.resolution == "atom":
            num_vertices = X.toc["atom"].sum(axis=-1).sum(axis=-1).ravel()
        else:
            num_vertices = X.toc["residue"].sum(axis=-1).ravel()
        num_molecules = len(num_vertices)
        vertex2batch = torch.arange(num_molecules).repeat_interleave(
            ak.to_torch(num_vertices)
        )
        if "molecule_graph" in X:
            offsets = np.insert(np.cumsum(num_vertices), 0, 0)[:-1]
            X.molecule_graph = X.molecule_graph + offsets
        return PygData(
            features=(
                ak.to_torch(X.vertex_features).float()
                if "vertex_features" in X
                else None
            ),
            token=(ak.to_torch(X.vertex_token).int() if "vertex_token" in X else None),
            pos=(ak.to_torch(X.vertex_pos).float() if "vertex_pos" in X else None),
            edge_index=(
                ak.to_torch(ak.concatenate(X.molecule_graph, axis=1)).long()
                if "molecule_graph" in X
                else None
            ),
            edge_attr=(
                ak.to_torch(ak.concatenate(X.molecule_graph_values, axis=0)).float()
                if "molecule_graph_values" in X
                else None
            ),
            # num_nodes=ak.to_torch(len(X[i]["vertices"].features)),
            mask=(ak.to_torch(X.vertex_mask).bool() if "vertex_mask" in X else None),
            y=(
                ak.to_torch(
                    ak.flatten(y["target"], axis=1)
                    if "sizes" in y.fields
                    else y["target"]
                ).float()
                if not y is None
                else None
            ),
            vertex2batch=vertex2batch,
            num_vertices=num_vertices,
            num_molecules=num_molecules,
            sizes=(y["sizes"] if not y is None and "sizes" in y.fields else None),
            **{attr: ak.to_torch(X.__getattr__(attr)) for attr in attr},
        )
