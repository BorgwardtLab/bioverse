import awkward as ak
import numpy as np

from ..task import Task
from ..utilities import PROTEIN_ALPHABET


class MaskedResiduePredictionTask(Task):
    def __init__(self):
        codes = np.char.encode(
            ak.ravel(list(PROTEIN_ALPHABET)).to_list(), "utf-8"
        ).view(np.uint8)
        self.lookup = np.zeros((256,), dtype=np.int32)
        self.lookup[codes] = np.arange(len(PROTEIN_ALPHABET)).astype(np.int64)

    def __call__(self, vbatch, assets, index):
        X = vbatch[index["scene"], index["frame"], index["molecule"]]
        X.resolution = "residue"
        targets = X.molecules.residue_label
        codes = np.char.encode(ak.ravel(targets).to_list(), "utf-8").view(np.uint8)
        tokens = self.lookup[codes]
        mask = ak.any(
            ak.local_index(X.molecules.residue_label) == index["residue"][:, None],
            axis=-1,
        )
        y = ak.Array({"target": ak.unflatten(tokens, ak.num(targets, axis=-1))[mask]})
        y["sizes"] = ak.num(y["target"], axis=-1)
        X.molecules.residue_mask = mask
        return X, y
