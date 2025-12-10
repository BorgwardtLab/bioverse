import awkward as ak
import numpy as np

from ..transform import Transform


class ResiduePositions(Transform):
    def __init__(self, mode="CA"):
        assert mode in ["CA", "COW", "C1'"]
        self.mode = mode

    def transform_batch(self, batch):
        if self.mode == "CA":
            # filter residues with missing atoms
            mask = ak.any(batch.residues.atom_label == "CA", axis=1)
            batch.residues = batch.residues[mask]
            batch.residue_pos = batch.atom_pos[batch.atom_label == "CA"]
        elif self.mode == "C1'":
            print(list(np.unique(ak.ravel(batch.atom_label))))
            print(
                ak.num(
                    batch.atom_pos[
                        np.logical_or(
                            batch.atom_label == "C1'", batch.atom_label == "C1"
                        )
                    ],
                    axis=0,
                )
            )
            print(ak.sum(batch.toc["atom"]))
            exit()
            batch.residue_pos = batch.atom_pos[batch.atom_label == "C1'"]
        elif self.mode == "COW":
            batch.residue_pos = batch.residues.atom_pos.mean(axis=1)
        return batch
