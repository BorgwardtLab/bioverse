import awkward as ak
import numpy as np
from scipy.spatial import cKDTree

from ..transform import Transform


class CoreAndSurfaceDistance(Transform):
    """
    RBF-encode per-residue distances to (a) the center-of-geometry of each
    chain/molecule and (b) the nearest surface residue, using only Cα atoms.

    Requires:
      - ``batch.residue_pos`` with shape [N_res, 3] containing Cα positions,
        e.g. from ``ResiduePositions(mode="CA")``.
      - ``batch.residues.residue_is_surface`` boolean mask from
        ``IsSurfaceResidue``.

    Produces:
      - ``batch.residues.residue_core_distance_rbf`` with shape
        [N_res, D_count]
      - ``batch.residues.residue_surface_distance_rbf`` with shape
        [N_res, D_count]
    """

    def __init__(
        self,
        D_min: float = 0.0,
        D_max: float = 30.0,
        D_count: int = 16,
    ):
        super().__init__()
        self.D_min = float(D_min)
        self.D_max = float(D_max)
        self.D_count = int(D_count)

    def _rbf(self, D: np.ndarray) -> np.ndarray:
        """
        Gaussian RBF expansion of distances.

        Parameters
        ----------
        D:
            1D array of distances [N].

        Returns
        -------
        np.ndarray
            RBF features with shape [N, D_count].
        """
        D_mu = np.linspace(self.D_min, self.D_max, self.D_count, dtype=float)
        D_mu = D_mu.reshape(1, -1)  # [1, D_count]
        D_sigma = (self.D_max - self.D_min) / max(self.D_count, 1)
        D_exp = D.reshape(-1, 1)  # [N, 1]
        return np.exp(-(((D_exp - D_mu) / D_sigma) ** 2))

    def transform_batch(self, batch):
        coords = batch.residue_pos.to_numpy()  # [N_tot, 3]
        N_tot = coords.shape[0]
        if N_tot == 0:
            empty = ak.Array(np.zeros((0, self.D_count), dtype=float))
            batch.residues.residue_core_distance_rbf = empty
            batch.residues.residue_surface_distance_rbf = empty
            return batch

        node_nums = np.asarray(batch.toc["residue"].ravel())
        if node_nums.size == 0:
            empty = ak.Array(np.zeros((0, self.D_count), dtype=float))
            batch.residues.residue_core_distance_rbf = empty
            batch.residues.residue_surface_distance_rbf = empty
            return batch

        offsets = np.concatenate([[0], np.cumsum(node_nums[:-1])])
        surface_mask_flat = np.asarray(
            batch.residues.residue_is_surface.to_numpy(), dtype=bool
        )
        core_dists = np.zeros((N_tot,), dtype=float)
        surf_dists = np.zeros((N_tot,), dtype=float)
        for count, start in zip(node_nums, offsets):
            n = int(count)
            if n <= 0:
                continue

            end = start + n
            P = coords[start:end]  # [n, 3]
            mask_surf = surface_mask_flat[start:end]

            # Distance to chain center-of-geometry
            center = P.mean(axis=0, keepdims=True)  # [1, 3]
            core_dists[start:end] = np.linalg.norm(P - center, axis=1)
            # Distance to nearest surface residue within this chain
            if np.any(mask_surf):
                P_surf = P[mask_surf]  # [n_s, 3]
                # Use a KD-tree for nearest-surface lookup to avoid O(n^2)
                # broadcasting when chains are long.
                tree = cKDTree(P_surf)
                dist, _ = tree.query(P, k=1, workers=-1)
                surf_dists[start:end] = dist
            else:
                # If no surface residues were identified, leave distances at 0.0
                surf_dists[start:end] = 0.0

        core_rbf = self._rbf(core_dists)
        surf_rbf = self._rbf(surf_dists)

        batch.residues.residue_core_distance_rbf = ak.Array(core_rbf)
        batch.residues.residue_surface_distance_rbf = ak.Array(surf_rbf)
        return batch
