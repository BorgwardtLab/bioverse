from __future__ import annotations

from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd

from ..adapter import Adapter
from ..data import Assets, Split
from ..transforms.structure_prediction import StructurePredictionTransform
from ..utilities import (
    PROTEIN_ALPHABET,
    IteratorWithLength,
    batched,
    config,
    download,
)

RAW_URL = (
    "https://github.com/BorgwardtLab/MLDEEP/releases/download/v1.0.0/raw.tar.gz"
)
# TEVp 0 catalytic domain; 1-indexed positions in single.tsv match this sequence.
WILDTYPE_PROTEASE = (
    "GESLFKGPRDYNPISSSICHLTNESDGHTTSLYGIGFGPFIITNKHLFRRNNGTLVVQSLHG"
    "VFKVKDTTTLQQHLVDGRDMIIIRMPKDFPPFPQKLKFREPQREERICLVTTNFQTKSMSSM"
    "VSDTSCTFPSGDGIFWKHWIQTKDGQCGSPLVSTRDGFIVGIHSASNFTNTNNYFTSVPKNF"
    "MELLTNQEAQQWVSGWRLNADSVLWGGHKVFMVKPEEPFQPVKEATQL"
)
WILDTYPE_SUBSTRATE = "ENLYFQS"
TEVS_INDEX = {"P6": 0, "P5": 1, "P4": 2, "P3": 3, "P2": 4, "P1": 5, "P1'": 6}
CANONICAL = set(PROTEIN_ALPHABET)


def _valid_mask(df: pd.DataFrame) -> pd.Series:
    tevp = df["TEVp"].astype(str)
    tevs = df["TEVs"].astype(str)
    mask = ~tevp.str.contains(r"\*", regex=True) & ~tevs.str.contains(
        r"\*", regex=True
    )
    protease_mut = df["n_mut_TEVp"] == 1
    substrate_mut = df["n_mut_TEVs"] == 1
    mask &= ~protease_mut | (
        df["TEVp_pos"].notna()
        & (df["TEVp_pos"] >= 1)
        & df["TEVp_AA"].isin(CANONICAL)
    )
    mask &= ~substrate_mut | (
        df["TEVs_pos"].isin(TEVS_INDEX) & df["TEVs_AA"].isin(CANONICAL)
    )
    return mask


def _sequence_record(protease: str, substrate: str) -> ak.Record:
    residue_label = ak.Array([list(protease), list(substrate)])
    residue_number = ak.Array(
        [
            np.arange(1, len(protease) + 1),
            np.arange(1, len(substrate) + 1),
        ]
    )
    data = {
        "chain_label": ak.Array(["A", "B"]),
        "residue_label": residue_label,
        "residue_number": residue_number,
    }
    data = {key: value[np.newaxis, np.newaxis] for key, value in data.items()}
    data["molecule_id"] = ak.Array([["TEVp_TEVs"]])
    return ak.Record(data)


def _find_single_tsv(root) -> pd.DataFrame:
    matches = list(root.rglob("single.tsv"))
    if not matches:
        raise FileNotFoundError(f"single.tsv not found under {root}.")
    return pd.read_csv(matches[0], sep="\t")


def mutation_tables(df: pd.DataFrame, protease: str = WILDTYPE_PROTEASE):
    """Return parallel mutation positions, labels, and AUC values."""
    df = df[_valid_mask(df)].reset_index(drop=True)
    protease_mut = df["n_mut_TEVp"].to_numpy() == 1
    substrate_mut = df["n_mut_TEVs"].to_numpy() == 1
    tevp_pos = df["TEVp_pos"].to_numpy()
    tevp_aa = df["TEVp_AA"].astype(str).to_numpy()
    tevs_pos = df["TEVs_pos"].astype(str).to_numpy()
    tevs_aa = df["TEVs_AA"].astype(str).to_numpy()
    auc = df["AUC"].to_numpy(dtype=float)
    offset = len(protease)
    positions, labels, effects = [], [], []
    for i in range(len(df)):
        pos, lab = [], []
        if protease_mut[i]:
            pos.append(int(tevp_pos[i]) - 1)
            lab.append(tevp_aa[i])
        if substrate_mut[i]:
            pos.append(offset + TEVS_INDEX[tevs_pos[i]])
            lab.append(tevs_aa[i])
        if pos:
            positions.append(pos)
            labels.append(lab)
            effects.append(float(auc[i]))
    return positions, labels, effects


class HuberProteaseAdapter(Adapter):
    """TEV protease–substrate single-mutant combinatorial screen from Huber et al."""

    @classmethod
    def download(
        cls,
        model: str = "boltz2",
        use_msa_server: bool = True,
        cache_dir: str | Path | None = None,
    ):
        path = config.raw_path / "MLDEEP"
        download(RAW_URL, path)
        df = _find_single_tsv(path)
        positions, labels, effects = mutation_tables(df, WILDTYPE_PROTEASE)

        record = _sequence_record(WILDTYPE_PROTEASE, WILDTYPE_SUBSTRATE)
        record = ak.Record(
            {
                **{field: record[field] for field in record.fields},
                "molecule_mutation_label": [[labels]],
                "molecule_mutation_pos": [[positions]],
                "molecule_mutation_effect": [[effects]],
            }
        )
        batches = batched(IteratorWithLength(iter([record]), 1))
        split = Split(
            {"huber_mutation_split": ["test"] * len(effects)},
            default="huber_mutation_split",
        )
        assets = Assets(
            {
                "protease_sequence": WILDTYPE_PROTEASE,
                "substrate_sequence": WILDTYPE_SUBSTRATE,
            }
        )
        transform = StructurePredictionTransform(
            model=model,
            use_msa_server=use_msa_server,
            cache_dir=cache_dir,
        )
        return transform(batches, split, assets)
