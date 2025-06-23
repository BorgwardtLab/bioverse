from pathlib import Path
from typing import Iterable

import awkward as ak
import numpy as np
from rdkit import Chem

from bioverse.processor import Processor
from bioverse.utilities import IteratorWithLength, progressbar


class SdfProcessor(Processor):

    valid_extensions = [".sdf"]

    @classmethod
    def process(cls, path: Path | str) -> Iterable[ak.Record]:
        path = Path(path)
        assert (
            path.suffix in cls.valid_extensions
        ), f'SdfProcessor does not support file extension "{path.suffix}"'
        processed = (
            cls.process_molecule(mol)
            for mol in progressbar(
                Chem.SDMolSupplier(str(path)), description="Processing"
            )
        )
        return IteratorWithLength(filter(lambda x: x is not None, processed))

    @classmethod
    def process_molecule(cls, molecule: Chem.Mol) -> ak.Record | None:
        try:
            smiles = Chem.MolToSmiles(molecule)
            atom_types = [atom.GetSymbol() for atom in molecule.GetAtoms()]
            coords = molecule.GetConformer().GetPositions().tolist()
            adj = Chem.GetAdjacencyMatrix(molecule)
            adj = np.array(adj.nonzero()).tolist()
        except Exception:
            return None
        data = {
            "molecule_id": [[molecule.GetProp("_Name").strip()]],
            "molecule_smiles": [[smiles]],
            "molecule_graph": [[adj]],
            "atom_types": [[[[atom_types]]]],  # todo: chemical groups
            "atom_pos": [[[[coords]]]],
        }
        return ak.Record(data)
