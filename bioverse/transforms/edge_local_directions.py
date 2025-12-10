import awkward as ak
import numpy as np

from ..transform import Transform


def _normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-7) -> np.ndarray:
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / (norm + eps)


class EdgeLocalDirections(Transform):
    """
    Per-edge local backbone directions expressed in the source residue frame,
    analogous to E_direct in the PiFold featuriser.

    Requires:
      - `batch.molecules.residue_backbone` [n_res, 4, 3] (N, CA, C, O)
      - `batch.molecules.residue_frame_R` [n_res, 3, 3] from `ResidueFrames`
      - `batch.molecules.molecule_edges` [n_edges, 2]

    Produces:
      - `batch.molecules.molecule_edge_direct` with shape [n_edges, 12] per
        molecule (4 unit vectors × 3 coordinates, flattened), where the 4
        vectors are from source CA to neighbor CA, N, C, O.
    """

    def transform_batch(self, batch):
        # Use flattened residue-level arrays (no molecule loop)
        backbone = (
            batch.residue_backbone
        )  # [N_tot, 4, 3] (possibly ragged in higher axes)
        R = batch.residue_frame_R  # [N_tot, 3, 3]

        backbone_np = np.asarray(backbone).reshape(-1, 4, 3)
        R_np = np.asarray(R).reshape(-1, 3, 3)

        N_tot = backbone_np.shape[0]
        if N_tot == 0:
            batch.molecules.molecule_edge_direct = ak.Array(
                [np.zeros((0, 12), dtype=float)]
            )
            return batch

        # Edge indices per molecule: [M, E_m, 2]
        edges = batch.molecules.molecule_edges
        edges_rel = ak.flatten(edges, axis=1)  # [E_tot, 2] (local indices)

        # Residues-per-molecule and edges-per-molecule counts
        res_counts = np.asarray(
            ak.to_numpy(ak.num(batch.molecules.residue_backbone, axis=1))
        )
        edge_counts = np.asarray(ak.to_numpy(ak.num(edges, axis=1)))

        if res_counts.size == 0 or edge_counts.size == 0:
            batch.molecules.molecule_edge_direct = ak.Array(
                [np.zeros((0, 12), dtype=float)]
            )
            return batch

        # Offsets to map local residue indices → global residue indices
        res_offsets = np.concatenate([[0], np.cumsum(res_counts[:-1])])
        edge_offsets = np.repeat(res_offsets, edge_counts)  # [E_tot]

        edges_rel_np = np.asarray(edges_rel, dtype=int)
        edges_global = edges_rel_np + edge_offsets[:, None]  # [E_tot, 2]

        src = edges_global[:, 0]
        dst = edges_global[:, 1]

        # Backbone atoms for all residues
        N = backbone_np[:, 0, :]
        CA = backbone_np[:, 1, :]
        C = backbone_np[:, 2, :]
        O = backbone_np[:, 3, :]

        # Vectors from source CA to neighbor CA, N, C, O
        dX = np.stack(
            [
                CA[dst] - CA[src],
                N[dst] - CA[src],
                C[dst] - CA[src],
                O[dst] - CA[src],
            ],
            axis=1,
        )  # [E_tot, 4, 3]

        # Express in local frame of source residue: v_local = R_src^T dX
        R_src_T = np.transpose(R_np[src], (0, 2, 1))  # [E_tot, 3, 3]
        v_local = np.einsum("eij,ekj->eki", R_src_T, dX)  # [E_tot, 4, 3]
        v_local = _normalize(v_local, axis=-1)

        feats_all = v_local.reshape(v_local.shape[0], -1)  # [E_tot, 12]

        # Unflatten back to per-molecule ragged structure
        feats_ragged = ak.unflatten(feats_all, edge_counts)
        batch.molecules.molecule_edge_direct = feats_ragged
        return batch
