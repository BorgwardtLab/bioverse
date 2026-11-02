import awkward as ak
import numpy as np

from ..collater import Collater


def _collate_targets(y):
    if y is None:
        return None, None
    if "sizes" in y.fields:
        return ak.flatten(y["target"], axis=1), y["sizes"]
    return y["target"], None


def _as_long_array(value):
    """Collapse leftover scene/frame nesting into a long tensor."""
    try:
        arr = np.asarray(ak.to_numpy(value))
    except Exception:
        return value
    squeeze = tuple(i for i, size in enumerate(arr.shape[:-1]) if size == 1)
    if squeeze:
        arr = np.squeeze(arr, axis=squeeze)
    # If the array is empty, avoid ambiguous `reshape(-1, 0)` behavior.
    # This can happen for optional metadata columns (e.g. some proteins
    # missing `molecule_enzyme_name` in a mixed shard batch).
    if arr.size == 0:
        return arr
    if arr.ndim <= 2:
        return arr
    if arr.shape[-2:] == (3, 3):
        return arr.reshape(-1, 3, 3)
    return arr.reshape(-1, arr.shape[-1])


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
        mutation_pos = None
        mutation_wt = None
        mutation_mut = None
        mutation_graph = None
        substrate_token = None
        if y is not None and "position" in y.fields:
            mutation_pos = np.asarray(ak.to_numpy(ak.ravel(y["position"])))
        if y is not None and "wt" in y.fields:
            mutation_wt = np.asarray(ak.to_numpy(ak.ravel(y["wt"])))
        if y is not None and "mut" in y.fields:
            mutation_mut = np.asarray(ak.to_numpy(ak.ravel(y["mut"])))
        if y is not None and "graph" in y.fields:
            mutation_graph = np.asarray(ak.to_numpy(ak.ravel(y["graph"])))
        if y is not None and "substrate" in y.fields:
            substrate_token = np.asarray(ak.to_numpy(y["substrate"]))
        node_attrs = {
            name: _as_long_array(X.__getattr__(name)) for name in attr if name in X
        }
        edge_attrs = {}
        for name in graphattr:
            if name in X:
                edge_attrs[name] = _as_long_array(
                    ak.concatenate(X.__getattr__(name), axis=0)
                )
        return Data(
            num_vertices=num_vertices,
            num_molecules=num_molecules,
            num_edges=num_edges if "molecule_edges" in X else None,
            vertex2molecule=vertex2molecule,
            edge2molecule=edge2molecule if "molecule_edges" in X else None,
            y=y_collated,
            mutation_pos=mutation_pos,
            mutation_wt=mutation_wt,
            mutation_mut=mutation_mut,
            mutation_graph=mutation_graph,
            substrate_token=substrate_token,
            _sizes=y_sizes,
            **node_attrs,
            **edge_attrs,
        )
