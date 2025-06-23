import awkward as ak

from ..task import Task


class StructurePredictionTask(Task):

    def __call__(self, vbatch, assets, index):
        X = vbatch[index["scene"], index["frame"], index["molecule"]]
        X.resolution = "residue"
        y = ak.Array(
            {
                "target": X.molecules.molecule_graph_distances,
                "sizes": ak.num(X.molecules.residue_label, axis=1),
                "residue_pos": X.molecules.residues.residue_pos,
                "residue_label": X.molecules.residues.residue_label,
            }
        )
        return X, y
