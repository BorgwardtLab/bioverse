import awkward as ak
import numpy as np

from ..transform import Transform


class ResidueBackboneAtoms(Transform):

    def transform_batch(self, batch):
        # filter residues with missing atoms
        mask = (
            ak.any(batch.residues.atom_label == "N", axis=1)
            & ak.any(batch.residues.atom_label == "CA", axis=1)
            & ak.any(batch.residues.atom_label == "C", axis=1)
            & ak.any(batch.residues.atom_label == "O", axis=1)
        )
        batch.residues = batch.residues[mask]
        # get N-CA-C-O coords
        N = batch.atom_pos[batch.atom_label == "N"]
        CA = batch.atom_pos[batch.atom_label == "CA"]
        C = batch.atom_pos[batch.atom_label == "C"]
        O = batch.atom_pos[batch.atom_label == "O"]
        assert len(N) == len(CA) == len(C) == len(O), "Some residues are missing atoms."
        batch.residues.residue_backbone = np.stack([N, CA, C, O], axis=1)
        return batch
