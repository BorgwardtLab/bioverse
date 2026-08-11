import awkward as ak
import numpy as np

from ..collater import Collater


def _collate_targets(y):
    if y is None:
        return None, None
    if "sizes" in y.fields:
        return ak.flatten(y["target"], axis=1), y["sizes"]
    return y["target"], None


class Data(object):

    """Container for long-format collated batch data."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if not value is None:
                setattr(self, key, value)

    def uncollate(self, y):
        if hasattr(self, "_sizes"):
            if hasattr(y, "numpy"):
                y = y.numpy()
            y = ak.unflatten(y, self._sizes, axis=0)
        y = ak.Array({"target": y})
        return y


class LongCollater(Collater):

    """Collate variable-length batches into padded long-format tensors."""

    @classmethod
    def collate(cls, X, y=None, attr=[], assets=None) -> Data:
        num_molecules = ak.sum(X.toc["molecule"].ravel())
        if X.resolution == "atom":
            num_vertices = X.toc["atom"].sum(axis=-1).sum(axis=-1).ravel()
            vertex2molecule = np.arange(num_molecules).repeat(num_vertices)
        else:
            num_vertices = X.toc["residue"].sum(axis=-1).ravel()
            vertex2molecule = np.arange(num_molecules).repeat(num_vertices)
        num_edges = None
        edge2molecule = None
        if "molecule_edges" in X:
            offsets = np.insert(np.cumsum(num_vertices), 0, 0)[:-1]
            X.molecules.molecule_edges = X.molecules.molecule_edges + offsets
            num_edges = X.molecules.molecule_edges.num(axis=1).ravel()
            edge2molecule = np.arange(num_molecules).repeat(num_edges)
        attr = [a for a in X.data.keys() if not a.startswith("molecule_edge")]
        graphattr = [a for a in X.data.keys() if a.startswith("molecule_edge")]
        y_collated, y_sizes = _collate_targets(y)
        return Data(
            num_vertices=num_vertices,
            num_molecules=num_molecules,
            num_edges=num_edges if "molecule_edges" in X else None,
            vertex2molecule=vertex2molecule,
            edge2molecule=edge2molecule if "molecule_edges" in X else None,
            y=y_collated,
            _sizes=y_sizes,
            **{name: X.__getattr__(name) for name in attr if name in X},
            **{
                name: ak.concatenate(X.__getattr__(name), axis=0)
                for name in graphattr
                if name in X
            },
        )
