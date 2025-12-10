import awkward as ak
import numpy as np

from ..transform import Transform


def _normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-7) -> np.ndarray:
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / (norm + eps)


class ResidueLocalDirections(Transform):
    """
    Per-residue local backbone directions (N, C, O) expressed in the residue
    local frame, analogous to V_direct in the PiFold featuriser.

    Requires:
      - `batch.residues.residue_backbone` with shape [n_res, 4, 3] (N, CA, C, O)
      - `batch.residues.residue_frame_T` and `batch.residues.residue_frame_R`
        from `ResidueFrames`

    Produces:
      - `batch.residues.residue_direct` with shape [n_res, 9] per batch
        (3 unit vectors × 3 coordinates, flattened).
    """

    def transform_batch(self, batch):
        backbone = batch.residues.residue_backbone
        T = batch.residues.residue_frame_T
        R = batch.residues.residue_frame_R

        backbone_np = np.asarray(backbone)
        T_np = np.asarray(T)
        R_np = np.asarray(R)

        # Vectors from CA (frame origin) to N, C, O
        N = backbone_np[:, 0, :]
        CA = backbone_np[:, 1, :]
        C = backbone_np[:, 2, :]
        O = backbone_np[:, 3, :]

        dX = np.stack([N - CA, C - CA, O - CA], axis=1)  # [N_res, 3, 3]

        # Express in local residue frame: v_local = R^T (dX)
        R_T = np.transpose(R_np, (0, 2, 1))
        v_local = np.einsum("nij,nkj->nki", R_T, dX)  # [N_res, 3, 3]
        v_local = _normalize(v_local, axis=-1)

        # Flatten 3 vectors × 3 coords
        residue_direct = v_local.reshape(v_local.shape[0], -1)
        batch.residues.residue_direct = ak.Array(residue_direct)
        return batch


