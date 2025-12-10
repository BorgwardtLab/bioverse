import awkward as ak
import numpy as np

from ..transform import Transform
from ..utilities import rotation_matrix_to_quaternion


class RelativeEdgeFramePositions(Transform):
    """
    Compute relative rotations between residue frames along edges, expressed as
    quaternions, analogous to E_angles in the PiFold featuriser but using the
    current edge set (`molecule_edges`) instead of a fixed linear graph.

    Requires:
      - `batch.molecules.residue_frame_R` [n_res, 3, 3] from `ResidueFrames`
      - `batch.molecules.molecule_edges` [n_edges, 2]

    Produces:
      - `batch.molecules.molecule_edge_quaternions` with shape [n_edges, 4]
        per molecule.
    """

    def transform_batch(self, batch):
        # Use flattened residue-frame array (no per-molecule Python loop)
        R_all = batch.residue_frame_R
        R_np = np.asarray(R_all).reshape(-1, 3, 3)  # [N_tot, 3, 3]

        N_tot = R_np.shape[0]
        if N_tot == 0:
            batch.molecules.molecule_edge_quaternions = ak.Array(
                [np.zeros((0, 4), dtype=float)]
            )
            return batch

        # Edge indices per molecule: [M, E_m, 2]
        edges = batch.molecules.molecule_edges
        edges_rel = ak.flatten(edges, axis=1)  # [E_tot, 2] (local indices)

        # Residues-per-molecule and edges-per-molecule counts
        res_counts = np.asarray(
            ak.to_numpy(ak.num(batch.molecules.residue_frame_R, axis=1))
        )
        edge_counts = np.asarray(ak.to_numpy(ak.num(edges, axis=1)))

        if res_counts.size == 0 or edge_counts.size == 0:
            batch.molecules.molecule_edge_quaternions = ak.Array(
                [np.zeros((0, 4), dtype=float)]
            )
            return batch

        # Offsets to map local residue indices → global residue indices
        res_offsets = np.concatenate([[0], np.cumsum(res_counts[:-1])])
        edge_offsets = np.repeat(res_offsets, edge_counts)  # [E_tot]

        edges_rel_np = np.asarray(edges_rel, dtype=int)
        edges_global = edges_rel_np + edge_offsets[:, None]  # [E_tot, 2]

        src = edges_global[:, 0]
        dst = edges_global[:, 1]

        # Frame rotations for all residues
        R_src = R_np[src]  # [E_tot, 3, 3]
        R_dst = R_np[dst]  # [E_tot, 3, 3]

        # Relative rotation from src to dst: R_rel = R_src^T * R_dst
        R_src_T = np.transpose(R_src, (0, 2, 1))
        R_rel = np.einsum("eij,ejk->eik", R_src_T, R_dst)  # [E_tot, 3, 3]

        quats_all = rotation_matrix_to_quaternion(R_rel)  # [E_tot, 4]

        # Unflatten back to per-molecule ragged structure
        quats_ragged = ak.unflatten(quats_all, edge_counts)
        batch.molecules.molecule_edge_quaternions = quats_ragged
        return batch
