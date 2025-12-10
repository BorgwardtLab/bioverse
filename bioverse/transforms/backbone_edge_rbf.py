import awkward as ak
import numpy as np

from ..transform import Transform


class BackboneEdgeRbf(Transform):
    """
    Compute RBF-expanded backbone distance features for multiple atom pairs along
    residue-residue edges, matching the pair-types used in the MPNN / PiFold
    featurisers.

    Requires:
      - `batch.molecules.residue_backbone` with shape [n_res, 4, 3] per molecule
        (N, CA, C, O) from `ResidueBackboneAtoms`
      - `batch.molecules.residue_cb` with shape [n_res, 3] per molecule from
        `ResiduePseudoCB`
      - `batch.molecules.molecule_edges` with shape [n_edges, 2] per molecule

    Produces:
      - `batch.molecules.molecule_edge_rbf_all` of shape
        [n_edges, num_pairs * D_count] per molecule.
    """

    def __init__(
        self,
        D_min: float = 2.0,
        D_max: float = 22.0,
        D_count: int = 16,
        feature_type: str = "mpnn",
    ):
        super().__init__()
        self.D_min = D_min
        self.D_max = D_max
        self.D_count = D_count
        assert feature_type in ("mpnn", "pifold")
        self.feature_type = feature_type

        if self.feature_type == "mpnn":
            # Pair list follows the original MPNN RBF_all ordering (25 pairs)
            self.pairs = [
                ("CA", "CA"),
                ("N", "N"),
                ("C", "C"),
                ("O", "O"),
                ("CB", "CB"),
                ("CA", "N"),
                ("CA", "C"),
                ("CA", "O"),
                ("CA", "CB"),
                ("N", "C"),
                ("N", "O"),
                ("N", "CB"),
                ("CB", "C"),
                ("CB", "O"),
                ("O", "C"),
                ("N", "CA"),
                ("C", "CA"),
                ("O", "CA"),
                ("CB", "CA"),
                ("C", "N"),
                ("O", "N"),
                ("CB", "N"),
                ("C", "CB"),
                ("O", "CB"),
                ("C", "O"),
            ]
        else:  # "pifold"
            # Pair list matches PiFold's 16 edge distance pairs (pair_lst)
            self.pairs = [
                ("CA", "CA"),
                ("CA", "C"),
                ("C", "CA"),
                ("CA", "N"),
                ("N", "CA"),
                ("CA", "O"),
                ("O", "CA"),
                ("C", "C"),
                ("C", "N"),
                ("N", "C"),
                ("C", "O"),
                ("O", "C"),
                ("N", "N"),
                ("N", "O"),
                ("O", "N"),
                ("O", "O"),
            ]

    def _rbf(self, D: np.ndarray) -> np.ndarray:
        D_mu = np.linspace(self.D_min, self.D_max, self.D_count)
        D_mu = D_mu.reshape(1, -1)
        D_sigma = (self.D_max - self.D_min) / self.D_count
        D = D[..., None]
        return np.exp(-(((D - D_mu) / D_sigma) ** 2))

    def _select_atom(
        self, backbone: np.ndarray, cb: np.ndarray, name: str
    ) -> np.ndarray:
        if name == "N":
            return backbone[:, 0, :]
        if name == "CA":
            return backbone[:, 1, :]
        if name == "C":
            return backbone[:, 2, :]
        if name == "O":
            return backbone[:, 3, :]
        if name == "CB":
            return cb
        raise ValueError(f"Unknown atom name '{name}'")

    def transform_batch(self, batch):
        # Backbone and pseudo-CB coordinates, flattened over higher axes but kept
        # as awkward arrays to avoid large explicit numpy conversions.
        backbone = batch.residue_backbone  # [N_tot, 4, 3] (awkward)
        cb = batch.residue_cb  # [N_tot, 3] (awkward)

        # Edge indices per molecule: [M, E_m, 2]
        edges = batch.molecules.molecule_edges
        edges_rel = ak.flatten(edges, axis=1)  # [E_tot, 2] (local indices, awkward)

        # Residues-per-molecule and edges-per-molecule counts (small, ok as numpy)
        res_counts = np.asarray(
            ak.to_numpy(ak.num(batch.molecules.residue_backbone, axis=1))
        )
        edge_counts = np.asarray(ak.to_numpy(ak.num(edges, axis=1)))

        if res_counts.size == 0 or edge_counts.size == 0:
            batch.molecules.molecule_edge_rbf_all = ak.Array(
                [np.zeros((0, len(self.pairs) * self.D_count))]
            )
            return batch

        # Offsets to map local residue indices → global residue indices
        res_offsets = np.concatenate([[0], np.cumsum(res_counts[:-1])])
        edge_offsets = ak.Array(np.repeat(res_offsets, edge_counts))  # [E_tot]

        # Convert local edge indices to global residue indices in awkward
        edges_global = edges_rel + edge_offsets[:, None]  # [E_tot, 2] (awkward)
        src = edges_global[:, 0]
        dst = edges_global[:, 1]

        E_tot = len(edges_rel)
        num_pairs = len(self.pairs)
        # Preallocate output instead of concatenating many arrays later
        feats_all = np.empty((E_tot, num_pairs * self.D_count), dtype=float)

        for pair_idx, (a_name, b_name) in enumerate(self.pairs):
            # Select atom coordinates per residue (awkward arrays)
            A = self._select_atom(backbone, cb, a_name)  # [N_tot, 3]
            B = self._select_atom(backbone, cb, b_name)  # [N_tot, 3]

            # Gather source/target atoms per edge and compute distances
            diff = A[src] - B[dst]  # [E_tot, 3], awkward
            D = np.sqrt(ak.sum(diff**2, axis=-1) + 1e-6)  # [E_tot], numpy
            rbf_feats = self._rbf(D)  # [E_tot, D_count], numpy
            start = pair_idx * self.D_count
            end = start + self.D_count
            feats_all[:, start:end] = rbf_feats

        # Unflatten back to per-molecule ragged structure
        feats_ragged = ak.unflatten(feats_all, edge_counts)
        batch.molecules.molecule_edge_rbf_all = feats_ragged
        return batch
