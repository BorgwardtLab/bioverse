import awkward as ak
import numpy as np

from ..task import Task
from ..utilities import index_put


class MutationEffectPredictionTask(Task):

    def __init__(self, index=0, resolution="residue") -> None:
        super().__init__()
        self.index = index
        self.resolution = resolution

    def __call__(self, vbatch, assets, index):
        X = vbatch[index["scene"], index["frame"], index["molecule"]]
        X.resolution = self.resolution
        amino_acid = X.molecule_mutation_labels[index["mutation"]]
        pos = X.molecule_mutation_pos[index["mutation"]]
        n = ak.local_index(amino_acid, axis=0).unflatten(1, -1)
        n, _ = ak.broadcast_arrays(n, amino_acid)
        n = n.unflatten(1, -1)
        pos, amino_acid = (
            ak.concatenate([n, pos], axis=2).flatten(axis=1),
            amino_acid.flatten(axis=1),
        )
        effects = X.molecule_mutation_effects[index["mutation"]][:, self.index]
        X.molecules.chains.residue_label = index_put(
            X.molecules.chains.residue_label, pos, amino_acid
        )
        y = ak.Array({"target": effects})
        return X, y

    """def __call__(self, vbatch, assets, index):
        X = vbatch[index["scene"], index["frame"], index["molecule"]]
        X.resolution = "residue"
        mutations = X.molecule_mutations[index["mutation"]]

        y = X.molecule_mutation_effect[index["mutation"]]
        targets = X.molecules.residue_label
        codes = np.char.encode(ak.ravel(targets).to_list(), "utf-8").view(np.uint8)
        tokens = self.lookup[codes]
        mask = ak.any(
            ak.local_index(X.molecules.residue_label) == index["residue"][:, None],
            axis=-1,
        )
        y = ak.Array({"target": ak.unflatten(tokens, ak.num(targets, axis=-1))[mask]})
        y.attrs["level"] = "molecule"
        return X, y"""
