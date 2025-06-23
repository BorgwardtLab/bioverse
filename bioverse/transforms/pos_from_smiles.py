import awkward as ak
import numpy as np
from sklearn.neighbors import kneighbors_graph

from ..transform import Transform


class GraphFromSmiles(Transform):

    def __init__(self, k=5, symmetric=True):
        super().__init__()
        self.k = k
        self.symmetric = symmetric

    def transform_batch(self, batch):
        edges = []
        for smiles in batch.smiles:  # type: ignore
            mol = Chem.MolFromSmiles(smiles)
            AllChem.Compute2DCoords(mol)
            atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
            coords = mol.GetConformer().GetPositions()[:, :2].tolist()
            adj = Chem.GetAdjacencyMatrix(mol)
            adj = np.array(adj.nonzero()).tolist()
            edges.append(edge_index)
        batch.molecule_graph = ak.Array(edges)
        return batch
