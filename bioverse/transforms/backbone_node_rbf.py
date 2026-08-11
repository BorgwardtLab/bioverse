import awkward as ak
import numpy as np

from ..transform import Transform


class BackboneNodeRbf(Transform):
    """Compute per-residue RBF-expanded within-residue backbone distance features.

    For each residue, computes Euclidean distances between the six intra-residue
    backbone atom pairs (Ca-N, Ca-C, Ca-O, N-C, N-O, O-C), expands each distance
    with a Gaussian RBF basis, and concatenates the result into
    ``6 * D_count`` node features per residue.

    Depends on ``batch.residues.residue_backbone`` from
    :class:`~bioverse.transforms.residue_backbone_atoms.ResidueBackboneAtoms`.
    Writes ``batch.residues.residue_node_rbf``.
    """

    def __init__(self, D_min: float = 0.0, D_max: float = 20.0, D_count: int = 16):
        super().__init__()
        self.D_min = D_min
        self.D_max = D_max
        self.D_count = D_count

        self.pairs = [
            ("CA", "N"),
            ("CA", "C"),
            ("CA", "O"),
            ("N", "C"),
            ("N", "O"),
            ("O", "C"),
        ]

    def _rbf(self, D: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian RBF expansion to distances.

        D: [N_res, P] distances (last axis = number of pairs, here P = 6)
        Returns: [N_res, P, D_count]
        """
        D_mu = np.linspace(self.D_min, self.D_max, self.D_count)
        D_mu = D_mu.reshape(1, 1, -1)  # [1,1,D_count] for broadcasting
        D_sigma = (self.D_max - self.D_min) / self.D_count
        D_exp = D[..., None]  # [N_res, P, 1]
        return np.exp(-(((D_exp - D_mu) / D_sigma) ** 2))

    def transform_batch(self, batch):
        # Backbone coordinates per residue (flatten higher axes): [N_res, 4, 3]
        backbone = batch.residues.residue_backbone
        backbone_np = np.asarray(backbone).reshape(-1, 4, 3)

        N_res_total = backbone_np.shape[0]
        if N_res_total == 0:
            batch.residues.residue_node_rbf = ak.Array(
                np.zeros((0, 6 * self.D_count), dtype=float)
            )
            return batch

        # Extract per-residue backbone atoms
        N = backbone_np[:, 0, :]  # [N_res, 3]
        CA = backbone_np[:, 1, :]
        C = backbone_np[:, 2, :]
        O = backbone_np[:, 3, :]

        # Compute the 6 within-residue distances for each residue
        d_CA_N = np.linalg.norm(CA - N, axis=-1)
        d_CA_C = np.linalg.norm(CA - C, axis=-1)
        d_CA_O = np.linalg.norm(CA - O, axis=-1)
        d_N_C = np.linalg.norm(N - C, axis=-1)
        d_N_O = np.linalg.norm(N - O, axis=-1)
        d_O_C = np.linalg.norm(O - C, axis=-1)

        D_pairs = np.stack(
            [d_CA_N, d_CA_C, d_CA_O, d_N_C, d_N_O, d_O_C], axis=-1
        )  # [N_res, 6]

        # RBF expansion → [N_res, 6, D_count], then flatten last two dims
        rbf = self._rbf(D_pairs)  # [N_res, 6, D_count]
        node_feats = rbf.reshape(N_res_total, -1)  # [N_res, 6*D_count]

        batch.residues.residue_node_rbf = ak.Array(node_feats)
        return batch
