import awkward as ak
import numpy as np

from ..task import Task
from ..utilities import index_put


def _mutation_table(scenes, field, row):
    table = getattr(scenes, field)[row]
    while table.ndim > 1 and len(table) == 1:
        table = table[0]
    return table


class MutationEffectPredictionTask(Task):

    def __init__(self, resolution="residue") -> None:
        super().__init__()
        self.resolution = resolution

    def __call__(self, vbatch, assets, index):
        X = vbatch[index["scene"], index["frame"], index["molecule"]]
        X.resolution = self.resolution
        mutations = ak.to_numpy(index["mutation"])
        amino_acid = ak.Array(
            [
                _mutation_table(X.scenes, "molecule_mutation_label", i)[int(m)]
                for i, m in enumerate(mutations)
            ]
        )
        pos = ak.Array(
            [
                _mutation_table(X.scenes, "molecule_mutation_pos", i)[int(m)]
                for i, m in enumerate(mutations)
            ]
        )
        n = ak.local_index(amino_acid, axis=0).unflatten(1, -1)
        n, _ = ak.broadcast_arrays(n, amino_acid)
        idx = ak.concatenate([n, pos], axis=1)
        effects = ak.Array(
            [
                float(_mutation_table(X.scenes, "molecule_mutation_effect", i)[int(m)])
                for i, m in enumerate(mutations)
            ]
        )
        X.molecules.residue_label = index_put(
            X.molecules.residue_label, idx, amino_acid.ravel()
        )
        y = ak.Array({"target": effects, "position": pos})
        return X, y
