import awkward as ak
import numpy as np

from ..transform import Transform


class IsSurfaceResidue(Transform):
    """
    Classify residues as surface or core using only Cα coordinates.

    This implements a simple, density-based rolling-ball style heuristic:
      - For each residue, we count how many other Cα atoms lie within a
        spherical neighborhood of radius ``probe_radius``.
      - Residues with comparatively few neighbors are marked as surface.

    Requires:
      - ``batch.residue_pos`` with shape [N_res, 3] containing Cα positions,
        e.g. from ``ResiduePositions(mode="CA")``.

    Produces:
      - ``batch.residues.residue_is_surface`` as a boolean array of shape
        [N_res], flattened over higher axes.
    """

    def __init__(
        self,
        probe_radius: float = 10.0,
        min_neighbors: int = 24,
        adaptive: bool = True,
        surface_fraction: float = 0.3,
    ):
        """
        Parameters
        ----------
        probe_radius:
            Neighborhood radius (in Å) for counting nearby Cα atoms.
        min_neighbors:
            Minimum neighbor count below which residues are considered surface.
        adaptive:
            If True, the threshold per chain is adapted so that at most
            ``surface_fraction`` of residues are classified as surface.
        surface_fraction:
            Target upper bound on the fraction of surface residues per chain
            when ``adaptive`` is True.
        """
        super().__init__()
        self.probe_radius = float(probe_radius)
        self.min_neighbors = int(min_neighbors)
        self.adaptive = bool(adaptive)
        self.surface_fraction = float(surface_fraction)

    def transform_batch(self, batch):
        coords = batch.residue_pos.to_numpy()  # [N_tot, 3]
        N_tot = coords.shape[0]
        if N_tot == 0:
            batch.residues.residue_is_surface = ak.Array(np.zeros((0,), dtype=bool))
            return batch

        # Residues-per-chain counts; flattened to 1D
        node_nums = np.asarray(batch.toc["residue"].ravel())
        if node_nums.size == 0:
            batch.residues.residue_is_surface = ak.Array(np.zeros((0,), dtype=bool))
            return batch

        offsets = np.concatenate([[0], np.cumsum(node_nums[:-1])])
        is_surface = np.zeros((N_tot,), dtype=bool)
        r2 = float(self.probe_radius) ** 2

        # Loop over chains/molecules (never over individual residues)
        for count, start in zip(node_nums, offsets):
            n = int(count)
            if n <= 0:
                continue
            end = start + n
            P = coords[start:end]  # [n, 3]

            # Pairwise squared distances within this chain: [n, n]
            diff = P[:, None, :] - P[None, :, :]
            dist2 = np.einsum("ijk,ijk->ij", diff, diff, optimize=True)

            neighbors = np.sum(dist2 <= r2, axis=1)

            if self.adaptive and n > 1:
                # Choose a per-chain threshold that keeps at most a certain
                # fraction of residues marked as surface, but never below
                # ``min_neighbors``.
                q = np.quantile(
                    neighbors,
                    max(0.0, min(1.0, 1.0 - self.surface_fraction)),
                )
                thresh = max(self.min_neighbors, int(round(q)))
            else:
                thresh = self.min_neighbors

            is_surface[start:end] = neighbors <= thresh

        batch.residues.residue_is_surface = ak.Array(is_surface)
        return batch
