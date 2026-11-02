from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
from biopandas.mmcif.engines import mmcif_col_types
from biopandas.mmcif.pandas_mmcif import load_cif_data

from ..processor import Processor
from ..utilities import THREE_TO_ONE


def _read_atom_df(path: Path) -> tuple[pd.DataFrame, str]:
    """Load ATOM rows from mmCIF, including Boltz files that omit auth_* columns."""
    if "".join(path.suffixes).lower().endswith(".gz"):
        import gzip

        with gzip.open(path, "rt") as handle:
            text = handle.read()
    else:
        text = path.read_text()
    data = load_cif_data(text)
    block = data[next(iter(data))]
    full = pd.DataFrame.from_dict(block["atom_site"], orient="index").transpose()
    if "auth_atom_id" not in full.columns and "label_atom_id" in full.columns:
        full["auth_atom_id"] = full["label_atom_id"]
    present = {col: dtype for col, dtype in mmcif_col_types.items() if col in full}
    full = full.astype(present, errors="ignore")
    return pd.DataFrame(full[full.group_PDB == "ATOM"]), text


class CifProcessor(Processor):
    """Parse mmCIF structure files into Awkward records."""

    valid_extensions = [".cif", ".cif.gz"]

    @classmethod
    def process_file(cls, path: str | Path, pLDDT: bool = False) -> ak.Record | None:

        # read cif file and metadata
        path = Path(path)
        df, pdb_text = _read_atom_df(path)
        is_alphafold = "AlphaFold" in pdb_text
        name = path.name.split(".")[0]

        # extract atom data to dict
        seq = pd.to_numeric(df["label_seq_id"], errors="coerce")
        if "auth_seq_id" in df.columns:
            seq = seq.where(seq.notna(), pd.to_numeric(df["auth_seq_id"], errors="coerce"))
        if seq.isna().any():
            filled = seq.copy()
            for _, index in df.groupby(df["label_asym_id"], sort=False).groups.items():
                vals = seq.loc[index]
                if vals.isna().all():
                    filled.loc[index] = np.arange(1, len(index) + 1)
                else:
                    filled.loc[index] = vals.ffill().bfill()
            seq = filled
        seq = seq.fillna(0).astype(int)
        residue_sizes = ak.Array(seq.to_numpy()).run_lengths()
        chain_sizes = ak.Array(df["label_asym_id"]).run_lengths()
        data = {
            "chain_label": df["label_asym_id"],
            "residue_number": np.asarray(seq, dtype=int),
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
