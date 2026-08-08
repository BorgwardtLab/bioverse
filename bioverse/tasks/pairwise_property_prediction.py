import awkward as ak
import numpy as np

from ..task import Task


class PairwisePropertyPredictionTask(Task):

    def __init__(
        self,
        meta: str,
        id_field: str = "id",
        level: str = "molecule",
        target: str | None = None,
        resolution: str = "residue",
    ) -> None:
        super().__init__()
        self.meta = meta
        self.id_field = id_field
        self.level = level
        self.target = target
        self.resolution = resolution
        self._matrices = {}
        self._interfaces = {}

    def _load_matrix(self, table):
        path = table["path"]
        if path not in self._matrices:
            self._matrices[path] = np.load(path, mmap_mode="r")
        return self._matrices[path]

    def _load_interfaces(self, table):
        from ..utilities import load

        path = table["path"]
        if path not in self._interfaces:
            self._interfaces[path] = load(path)
        return self._interfaces[path]

    def _lookup_interface_contacts(self, table, id1, id2):
        interfaces = self._load_interfaces(table)
        chain_lengths = table["chain_lengths"]
        pdbid, chain1 = id1.split("_", 1)
        _, chain2 = id2.split("_", 1)
        n1 = chain_lengths[id1]
        n2 = chain_lengths[id2]
        contacts = np.zeros(n1 * n2, dtype=np.float32)
        inds = interfaces.get(pdbid, {}).get(chain1, {}).get(chain2, [])
        if inds:
            inds = np.asarray(inds, dtype=int)
            contacts[inds[:, 0] * n2 + inds[:, 1]] = 1.0
        return contacts.tolist()

    def _lookup_target(self, table, id1, id2):
        if "path" in table and "chain_lengths" in table:
            return self._lookup_interface_contacts(table, id1, id2)
        if "path" in table and "index" in table:
            matrix = self._load_matrix(table)
            value = float(matrix[table["index"][id1], table["index"][id2]])
            if "standardize_mean" in table and "standardize_std" in table:
                value = (value - table["standardize_mean"]) / table["standardize_std"]
            return value
        value = table[id1][id2]
        if self.level == "residue" and self.target is not None:
            value = value[self.target]
        return value

    def __call__(self, vbatch, assets, index):
        X1 = vbatch[index["scene"], index["frame"], index["molecule"]]
        X2 = vbatch[index["scene2"], index["frame2"], index["molecule2"]]
        X1.resolution = self.resolution
        X2.resolution = self.resolution
        X = (X1, X2)

        id_field = f"{self.level}_{self.id_field}"
        table = assets[self.meta]
        ids1 = ak.to_list(ak.ravel(getattr(X1.molecules, id_field)))
        ids2 = ak.to_list(ak.ravel(getattr(X2.molecules, id_field)))
        targets = [
            self._lookup_target(table, id1, id2) for id1, id2 in zip(ids1, ids2)
        ]

        if self.level == "molecule":
            y = ak.Array({"target": targets})
        else:
            sizes = [len(target) for target in targets]
            y = ak.Array({"target": ak.unflatten(targets, sizes), "sizes": sizes})
        return X, y
