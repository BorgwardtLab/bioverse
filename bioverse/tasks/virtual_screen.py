import awkward as ak

from ..task import Task


class VirtualScreenTask(Task):

    def __call__(self, vbatch, assets, index):
        X = vbatch[index["scene"], index["frame"], index["molecule"]]
        targets = []
        for ligands, decoys in zip(
            ak.to_list(ak.ravel(X.molecule_ligands_smiles)),
            ak.to_list(ak.ravel(X.molecule_decoys_smiles)),
        ):
            targets.append([1] * len(ligands) + [0] * len(decoys))
        return X, ak.Array({"target": targets})
