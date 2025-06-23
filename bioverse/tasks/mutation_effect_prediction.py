import awkward as ak

from ..task import Task


class MutationEffectPredictionTask(Task):

    def __call__(self, vbatch, assets, index):
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
        return X, y
