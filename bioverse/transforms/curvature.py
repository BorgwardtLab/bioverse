import awkward as ak
import numpy as np

from ..transform import Transform


class Curvature(Transform):
    """
    Approximate backbone curvature at each residue from Cα positions and
    mask it to surface residues.

    Curvature is estimated per chain from triples of consecutive Cα atoms
    (p_{i-1}, p_i, p_{i+1}) as the inverse radius of the circle through the
    three points:

        κ_i = 2 * ||(p_i - p_{i-1}) × (p_{i+1} - p_i)|| /
              (||p_i - p_{i-1}|| * ||p_{i+1} - p_i|| * ||(p_{i+1} - p_{i-1})||)

    Endpoints (residues without both neighbors) are assigned zero curvature.

    Requires:
      - ``batch.residue_pos`` with shape [N_res, 3] containing Cα positions,
        e.g. from ``ResiduePositions(mode="CA")``.
      - ``batch.residues.residue_is_surface`` boolean mask from
        ``IsSurfaceResidue``.

    Produces:
      - ``batch.residues.residue_curvature`` with shape [N_res], where
        non-surface residues are set to ``default_value``.
    """

    def __init__(self, default_value: float = 0.0, eps: float = 1e-7):
        super().__init__()
        self.default_value = float(default_value)
        self.eps = float(eps)

    def transform_batch(self, batch):
        coords = batch.residue_pos.to_numpy()  # [N_tot, 3]
        N_tot = coords.shape[0]
        if N_tot == 0:
            batch.residues.residue_curvature = ak.Array(np.zeros((0,), dtype=float))
            return batch

        node_nums = np.asarray(batch.toc["residue"].ravel())
        if node_nums.size == 0:
            batch.residues.residue_curvature = ak.Array(np.zeros((0,), dtype=float))
            return batch

        offsets = np.concatenate([[0], np.cumsum(node_nums[:-1])])
        curvature = np.full((N_tot,), self.default_value, dtype=float)

        for count, start in zip(node_nums, offsets):
            n = int(count)
            if n < 3:
                continue

            end = start + n
            P = coords[start:end]  # [n, 3]

            # Vectors between consecutive residues
            u = P[1:-1] - P[:-2]  # [n-2, 3]
            v = P[2:] - P[1:-1]  # [n-2, 3]
            w = P[2:] - P[:-2]  # [n-2, 3]

            cross = np.cross(u, v)
            area2 = np.linalg.norm(cross, axis=-1)  # 2 * area of triangle

            du = np.linalg.norm(u, axis=-1)
            dv = np.linalg.norm(v, axis=-1)
            dw = np.linalg.norm(w, axis=-1)

            denom = du * dv * dw
            kappa = 2.0 * area2 / (denom + self.eps)  # [n-2]

            # Assign to internal residues of this chain
            curvature[start + 1 : end - 1] = kappa

        surface_mask_flat = np.asarray(
            batch.residues.residue_is_surface.to_numpy(), dtype=bool
        )
        curvature[~surface_mask_flat] = self.default_value

        batch.residues.residue_curvature = ak.Array(curvature)
        return batch
