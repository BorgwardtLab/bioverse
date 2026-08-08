import awkward as ak
import numpy as np

from ..task import Task
from ..utilities import index_put


class MutationEffectPredictionTask(Task):

    def __init__(self, resolution="residue") -> None:
        super().__init__()
        self.resolution = resolution

    def __call__(self, vbatch, assets, index):
        X = vbatch[index["scene"], index["frame"], index["molecule"]]
        X.resolution = self.resolution
        _ = np.arange(len(index["mutation"]))
        amino_acid = X.scenes.molecule_mutation_label[_, index["mutation"]]
        pos = X.scenes.molecule_mutation_pos[_, index["mutation"]]
        # assert ak.all(ak.num(X.molecules.residue_label, axis=1) > pos.ravel())
        n = ak.local_index(amino_acid, axis=0).unflatten(1, -1)
        n, _ = ak.broadcast_arrays(n, amino_acid)
        # n = n.unflatten(1, -1)
        idx = ak.concatenate([n, pos], axis=1)
        effects = X.molecule_mutation_effect[index["mutation"]]
        X.molecules.residue_label = index_put(
            X.molecules.residue_label, idx, amino_acid.ravel()
        )
        y = ak.Array({"target": effects, "position": pos})
        return X, y
