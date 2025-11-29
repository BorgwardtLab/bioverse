import awkward as ak
import numpy as np

from ..transform import Transform
from ..utilities import normalize


class Orientations(Transform):

    def transform_batch(self, batch):
        # From https://github.com/drorlab/gvp-pytorch.git
        mask = ak.any(batch.residues.atom_label == "CA", axis=1)
        batch.residues = batch.residues[mask]
        CA = batch.atom_pos[batch.atom_label == "CA"].to_numpy()
        breakpoints = np.cumsum(batch.toc["residue"].ravel()) - 1
        forward = normalize(CA[1:] - CA[:-1])
        backward = normalize(CA[:-1] - CA[1:])
        forward = np.pad(forward, ((0, 1), (0, 0)))
        forward[breakpoints] = 0.0
        backward = np.pad(backward, ((1, 0), (0, 0)))
        backward[np.concatenate([[0], breakpoints[:-1]])] = 0.0
        orientations = np.concatenate(
            [np.expand_dims(forward, -2), np.expand_dims(backward, -2)], -2
        )
        batch.residue_orientations = orientations
        return batch
