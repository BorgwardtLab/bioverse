import awkward as ak
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from ..transform import Transform


class GraphFromSmiles(Transform):
    filter = "scenes"

    def __init__(self, geometric=False, dim=2, hydrogens=False):
        assert dim in [2, 3], "dim must be 2 or 3"
        super().__init__()
        self.geometric = geometric
        self.dim = dim
        self.hydrogens = hydrogens

    def transform_batch(self, batch):
        graph, pos, labels, mask = [], [], [], []
        for smiles in batch.molecule_smiles:  # type: ignore
            try:
                mol = Chem.MolFromSmiles(smiles)
                if self.hydrogens:
                    mol = Chem.AddHs(mol)
                if self.geometric:
                    if self.dim == 2:
                        AllChem.Compute2DCoords(mol)
                        coords = mol.GetConformer().GetPositions()[:, :2].tolist()
                    elif self.dim == 3:
                        if not self.hydrogens:  # H are necessary for embedding
                            mol = Chem.AddHs(mol)
                        AllChem.EmbedMolecule(mol, randomSeed=42)
                        AllChem.MMFFOptimizeMolecule(mol)
                        if not self.hydrogens:
                            mol = Chem.RemoveHs(mol)
                        coords = mol.GetConformer().GetPositions().tolist()
                    pos.append(coords)
                adj = Chem.GetAdjacencyMatrix(mol)
                adj = np.array(adj.nonzero()).T.tolist()
                mol_labels = [atom.GetSymbol() for atom in mol.GetAtoms()]
                graph.append(adj)
                mask.append(True)
                labels.append(mol_labels)
            except AssertionError:
                raise
            except:
                if self.geometric:
                    pos.append([None])
                graph.append([None])
                mask.append(False)
                labels.append([None])
        batch.molecule_edges = ak.Array(graph)
        batch.molecules.atom_label = ak.Array(labels)
        if self.geometric:
            batch.molecules.atom_pos = ak.Array(pos)
        batch.scene_filter = ak.Array(mask)
        return batch
