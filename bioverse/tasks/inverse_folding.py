import awkward as ak
import numpy as np

from ..task import Task
from ..utilities import PROTEIN_ALPHABET


class InverseFoldingTask(Task):
    def __init__(self, resolution="residue"):
        codes = np.char.encode(
            ak.ravel(list(PROTEIN_ALPHABET)).to_list(), "utf-8"
        ).view(np.uint8)
        self.resolution = resolution
        self.lookup = np.zeros((256,), dtype=np.int32)
        self.lookup[codes] = np.arange(len(PROTEIN_ALPHABET)).astype(np.int64)

    def __call__(self, vbatch, assets, index):
        X = vbatch[index["scene"], index["frame"], index["molecule"]]
        X.resolution = self.resolution
        targets = X.molecules.residue_label
        sizes = ak.num(targets, axis=-1)
        codes = np.char.encode(ak.ravel(targets).to_list(), "utf-8").view(np.uint8)
        tokens = self.lookup[codes]
        y = ak.Array({"target": ak.unflatten(tokens, sizes)})
        y["sizes"] = sizes
        if self.resolution == "atom":
            X.molecules.atom_mask = X.molecules.atom_label == "CA"
        return X, y
