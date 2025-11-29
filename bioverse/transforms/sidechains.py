import awkward as ak
import numpy as np

from ..transform import Transform
from ..utilities import normalize


class Sidechains(Transform):

    def transform_batch(self, batch):
        # From https://github.com/drorlab/gvp-pytorch.git

        # filter residues with missing atoms
        mask = (
            ak.any(batch.residues.atom_label == "N", axis=1)
            & ak.any(batch.residues.atom_label == "CA", axis=1)
            & ak.any(batch.residues.atom_label == "C", axis=1)
        )
        batch.residues = batch.residues[mask]

        N = batch.atom_pos[batch.atom_label == "N"]
        CA = batch.atom_pos[batch.atom_label == "CA"]
        C = batch.atom_pos[batch.atom_label == "C"]
        assert len(N) == len(CA) == len(C), "Some residues are missing atoms."

        c, n = normalize(C - CA), normalize(N - CA)
        bisector = normalize(c + n)
        perp = normalize(np.cross(c, n))
        sidechains = -bisector * np.sqrt(1 / 3) - perp * np.sqrt(2 / 3)
        batch.residue_sidechains = sidechains
        return batch
