from pathlib import Path

import awkward as ak
import numpy as np
from biopandas.pdb import PandasPdb
from biopandas.pdb.engines import amino3to1dict

from ..processor import Processor


class PdbProcessor(Processor):

    valid_extensions = [".pdb", ".pdb.gz"]

    @classmethod
    def process_file(cls, path: str | Path, pLDDT: bool = False) -> ak.Record:

        # read pdb file and metadata
        path = Path(path)
        pdb = PandasPdb().read_pdb(str(path))
        is_alphafold = (
            pdb.df["OTHERS"]["entry"].str.contains("ALPHAFOLD", case=False).any()
        )
        name = path.name.split(".")[0]

        # extract atom data to dict
        df = pdb.df["ATOM"]
        residue_sizes = ak.Array(df["residue_number"].to_numpy()).run_lengths()
        chain_sizes = ak.Array(df["chain_id"].to_list()).run_lengths()
        data = {
            "chain_label": df["chain_id"].to_list(),
            "residue_number": df["residue_number"].to_numpy(),
            "residue_label": df["residue_name"].map(amino3to1dict).to_list(),
            "atom_label": df["atom_name"].to_list(),
            "atom_pos": df[["x_coord", "y_coord", "z_coord"]].to_numpy(),
            "atom_b_factor": df["b_factor"].to_numpy(),
        }

        # add nesting for chains and residues
        data = {
            k: ak.Array(v).unflatten(chain_sizes).unflatten(residue_sizes, 1)
            for k, v in data.items()
        }

        # atom-wise labels to residue- or chain-wise labels
        data["residue_number"] = data["residue_number"].firsts(2)
        data["residue_label"] = data["residue_label"].firsts(2)
        data["chain_label"] = data["chain_label"].firsts(2).firsts(1)

        # add pLDDT if available
        if is_alphafold or pLDDT:
            data["residue_pLDDT"] = data.pop("atom_b_factor").firsts()

        # reshape to scenes of frames of molecules
        data = {k: v[np.newaxis, np.newaxis] for k, v in data.items()}

        # add molecule attributes
        data["molecule_id"] = ak.Array([[name]])

        return ak.Record(data)
