from pathlib import Path

import awkward as ak
import numpy as np
from biopandas.mmcif import PandasMmcif

from ..processor import Processor
from ..utilities import THREE_TO_ONE


class CifProcessor(Processor):
    valid_extensions = [".cif", ".cif.gz"]

    @classmethod
    def process_file(cls, path: str | Path, pLDDT: bool = False) -> ak.Record | None:

        # read cif file and metadata
        path = Path(path)
        mmcif = PandasMmcif().read_mmcif(str(path))
        df = mmcif.df["ATOM"]
        is_alphafold = "AlphaFold" in mmcif.pdb_text
        name = path.name.split(".")[0]

        # extract atom data to dict
        residue_sizes = ak.Array(df["label_seq_id"]).run_lengths()
        chain_sizes = ak.Array(df["label_asym_id"]).run_lengths()
        data = {
            "chain_label": df["label_asym_id"],
            "residue_number": np.array(df["label_seq_id"]).astype(int),
            "residue_label": df["label_comp_id"],
            "atom_label": df["label_atom_id"],
            "atom_pos": np.stack(
                (
                    np.array(df["Cartn_x"], dtype=np.float32),
                    np.array(df["Cartn_y"], dtype=np.float32),
                    np.array(df["Cartn_z"], dtype=np.float32),
                ),
                axis=1,
            ),
            "atom_b_factor": np.array(df["B_iso_or_equiv"]).astype(np.float32),
        }
        data["residue_label"] = [
            THREE_TO_ONE[residue] if residue in THREE_TO_ONE else residue
            for residue in data["residue_label"]
        ]

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
