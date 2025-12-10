import awkward as ak
import numpy as np

from ..data import Batch
from ..transform import Transform


class CreateFeatureFromOthers(Transform):
    """
    Concatenate existing attributes into a single feature tensor.

    Parameters
    ----------
    feature_level : {"vertex", "edge"}
        Whether to create a vertex-level (per residue / atom) or edge-level
        (per edge) feature.
    source_attrs : list[str]
        List of attribute names to concatenate, e.g.
        - vertex level: ["residue_dihedrals", "residue_bond_angles"]
        - edge level:   ["molecule_edge_rbf", "molecule_edge_vectors"]
    target_attr : str
        Name of the new attribute to write into the batch.
        The correct prefix is inferred from `feature_level`:
        - "vertex": prefix "vertex" is mapped to "residue" or "atom"
        - "edge":   you should pass a full name, e.g. "molecule_edge_features"
    """

    def __init__(
        self,
        feature_level: str,
        source_attrs: list[str],
        target_attr: str,
    ):
        assert feature_level in ("vertex", "edge")
        self.feature_level = feature_level
        self.source_attrs = source_attrs
        self.target_attr = target_attr

    def _get_vertex_array(self, batch: Batch, name: str) -> ak.Array:
        # Use the BatchProxy vertex aliasing where possible
        return getattr(batch, name)

    def _get_vertex_array_for_edges(self, batch: Batch, name: str) -> ak.Array:
        # For edge-level broadcasting, work at molecule level so indexing
        # aligns with per-molecule edge indices.
        return getattr(batch.molecules, name)

    def _get_edge_array(self, batch: Batch, name: str) -> ak.Array:
        # Edge attributes live under the "molecule" prefix in current transforms
        return getattr(batch.molecules, name)

    def _broadcast_vertex_to_edges(self, v: ak.Array, edge_index: ak.Array) -> ak.Array:
        """
        Broadcast vertex features to edges using awkward indexing.

        v: [M, N_m, Fv] vertex features per molecule
        edge_index: [M, E_m, 2] edge indices per molecule

        Returns:
            edge_features: [M, E_m, Fv]
        """
        # Destination node indices per edge
        dst = edge_index[..., 1]
        # Advanced indexing over the second axis: per-molecule gather
        edge_feats = v[dst]
        return edge_feats

    def transform_batch(self, batch: Batch) -> Batch:
        if self.feature_level == "vertex":
            # All attributes must be vertex-level; fetch and concatenate
            arrays: list[ak.Array] = []
            for name in self.source_attrs:
                v = self._get_vertex_array(batch, name)
                arrays.append(v)

            if not arrays:
                return batch

            # Convert to dense numpy, treating scalar features as width-1 vectors.
            np_arrays: list[np.ndarray] = []
            for a in arrays:
                arr = ak.to_numpy(a)
                if arr.ndim == 1:
                    arr = arr[:, None]
                np_arrays.append(arr)

            N = np_arrays[0].shape[0]
            feat_dim = sum(a.shape[1] for a in np_arrays)
            features_np = np.empty((N, feat_dim), dtype=np_arrays[0].dtype)

            offset = 0
            for a in np_arrays:
                width = a.shape[1]
                features_np[:, offset : offset + width] = a
                offset += width

            features = ak.Array(features_np)

            # Write as a vertex-level attribute: "vertex_*" alias is resolved by Batch
            setattr(batch, self.target_attr, features)

        elif self.feature_level == "edge":
            # Need access to edge_index to broadcast any vertex attributes
            edge_index = batch.molecules.molecule_edges
            edge_arrays: list[ak.Array] = []

            for name in self.source_attrs:
                # Heuristic: if name starts with "vertex" or "residue"/"atom",
                # treat as vertex-level; otherwise as edge-level.
                if (
                    name.startswith("vertex")
                    or name.startswith("residue")
                    or name.startswith("atom")
                ):
                    v = self._get_vertex_array_for_edges(batch, name)
                    edge_v = self._broadcast_vertex_to_edges(v, edge_index)
                    edge_arrays.append(edge_v)
                else:
                    e = self._get_edge_array(batch, name)
                    edge_arrays.append(e)

            if edge_arrays:
                # All edge arrays share the same ragged edge structure; flatten,
                # pre-allocate dense storage, then unflatten back.
                edge_counts = ak.num(edge_index, axis=1)
                flat_arrays_np: list[np.ndarray] = []
                for a in edge_arrays:
                    fa = ak.to_numpy(ak.flatten(a, axis=1))
                    if fa.ndim == 1:
                        fa = fa[:, None]
                    flat_arrays_np.append(fa)

                E_tot = flat_arrays_np[0].shape[0]
                feat_dim = sum(a.shape[1] for a in flat_arrays_np)
                features_np = np.empty((E_tot, feat_dim), dtype=flat_arrays_np[0].dtype)

                offset = 0
                for a in flat_arrays_np:
                    width = a.shape[1]
                    features_np[:, offset : offset + width] = a
                    offset += width

                features = ak.unflatten(features_np, ak.to_numpy(edge_counts))
                setattr(batch.molecules, self.target_attr, features)

        return batch
