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

SCORES_URL = (
    "https://media.githubusercontent.com/media/jbloomlab/"
    "SARS-CoV-2-RBD_DMS_variants/main/results/final_variant_scores/"
    "final_variant_scores.csv"
)
# Spike 331–531; matches data/wildtype_sequence.fasta and RBD_sites.csv.
WILDTYPE_RBD = (
    "NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVY"
    "ADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNL"
    "KPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCG"
    "PKKST"
)
# Human ACE2 ectodomain from 6m0j SEQRES chain A, without the C-terminal His-tag
# (UniProt Q9BYF1 residues 19–615).
WILDTYPE_ACE2 = (
    "STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMY"
    "PLQEIQNLTVKLQLQALQQNGSSVLSEDKSKRLNTILNTMSTIYSTGKVCNPDNPQECLLLEPGL"
    "NEIMANSLDYNERLWAWESWRSEVGKQLRPLYEEYVVLKNEMARANHYEDYGDYWRGDYEVNGVD"
    "GYDYSRGQLIEDVEHTFEEIKPLYEHLHAYVRAKLMNAYPSYISPIGCLPAHLLGDMWGRFWTNL"
    "YSLTVPFGQKPNIDVTDAMVDQAWDAQRIFKEAEKFFVSVGLPNMTQGFWENSMLTDPGNVQKAV"
    "CHPTAWDLGKGDFRILMCTKVTMDDFLTAHHEMGHIQYDMAYAAQPFLLRNGANEGFHEAVGEIM"
    "SLSAATPKHLKSIGLLSPDFQEDNETEINFLLKQALTIVGTLPFTYMLEKWRWMVFKGEIPKDQW"
    "MKKWWEMKREIVGVVEPVPHDETYCDPASLFHVSNDYSFIRYYTRTLYQFQFQEALCQAAKHEGP"
    "LHKCDISNSTEAGQKLFNMLRLGKSEPWTLALENVVGAKNMNVRPLLNYFEPLFTWLKDQNKNSF"
    "VGWSTDWSPYAD"
)
BACKGROUND = "Wuhan-Hu-1"
SITE_OFFSET = 331
CANONICAL = set(PROTEIN_ALPHABET)


def _valid_mask(
    df: pd.DataFrame,
    rbd: str = WILDTYPE_RBD,
    target: str = BACKGROUND,
) -> pd.Series:
    last = SITE_OFFSET + len(rbd) - 1
    wt = df["wildtype"].astype(str)
    mut = df["mutant"].astype(str)
    pos = df["position"]
    in_range = pos.between(SITE_OFFSET, last)
    seq_wt = pd.Series(pd.NA, index=df.index, dtype=object)
    seq_wt.loc[in_range] = [rbd[int(p) - SITE_OFFSET] for p in pos[in_range]]
    return (
        (df["target"] == target)
        & (wt != mut)
        & wt.isin(CANONICAL)
        & mut.isin(CANONICAL)
        & in_range
        & (wt == seq_wt)
        & df["delta_bind"].notna()
    )


def _sequence_record(rbd: str, ace2: str) -> ak.Record:
    residue_label = ak.Array([list(rbd), list(ace2)])
    residue_number = ak.Array(
        [
            np.arange(SITE_OFFSET, SITE_OFFSET + len(rbd)),
            np.arange(19, 19 + len(ace2)),
        ]
    )
    data = {
        "chain_label": ak.Array(["A", "B"]),
        "residue_label": residue_label,
        "residue_number": residue_number,
    }
    data = {key: value[np.newaxis, np.newaxis] for key, value in data.items()}
    data["molecule_id"] = ak.Array([["RBD_ACE2"]])
    return ak.Record(data)


def _find_scores_csv(root) -> pd.DataFrame:
    matches = list(Path(root).rglob("final_variant_scores.csv"))
    if not matches:
        raise FileNotFoundError(f"final_variant_scores.csv not found under {root}.")
    return pd.read_csv(matches[0])


def mutation_tables(
    df: pd.DataFrame,
    rbd: str = WILDTYPE_RBD,
    target: str = BACKGROUND,
):
    """Return parallel RBD mutation positions, labels, and Δlog10Ka values."""
    df = df[_valid_mask(df, rbd, target)].reset_index(drop=True)
    positions = [[int(pos) - SITE_OFFSET] for pos in df["position"]]
    labels = [[str(aa)] for aa in df["mutant"]]
    effects = [float(value) for value in df["delta_bind"]]
    return positions, labels, effects


class StarrRbdAdapter(Adapter):
    """SARS-CoV-2 RBD DMS in the Wuhan-Hu-1 background from Starr et al."""

    @classmethod
    def download(
        cls,
        model: str = "boltz2",
        use_msa_server: bool = True,
        cache_dir: str | Path | None = None,
    ):
        path = config.raw_path / "SARS_CoV_2_RBD_DMS"
        download(SCORES_URL, path / "final_variant_scores.csv")
        df = _find_scores_csv(path)
        positions, labels, effects = mutation_tables(df, WILDTYPE_RBD)

        record = _sequence_record(WILDTYPE_RBD, WILDTYPE_ACE2)
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
            {"starr_mutation_split": ["test"] * len(effects)},
            default="starr_mutation_split",
        )
        assets = Assets(
            {
                "rbd_sequence": WILDTYPE_RBD,
                "ace2_sequence": WILDTYPE_ACE2,
            }
        )
        transform = StructurePredictionTransform(
            model=model,
            use_msa_server=use_msa_server,
            cache_dir=cache_dir,
        )
        return transform(batches, split, assets)
