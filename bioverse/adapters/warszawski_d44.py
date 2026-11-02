from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

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
    "https://journals.plos.org/ploscompbiol/article/file"
    "?id=10.1371/journal.pcbi.1007207.s010&type=supplementary"
)
# Variable domains from PDB 1MLC (Fv only; constant regions omitted).
WILDTYPE_VH = (
    "QVQLQESGAEVMKPGASVKISCKATGYTFSTYWIEWVKQRPGHGLEWIGEILPGSGSTYYNEKFK"
    "GKATFTADTSSNTAYMQLSSLTSEDSAVYYCARGDGNYGYWGQGTTLTVSS"
)
WILDTYPE_VL = (
    "DIELTQSPATLSVTPGDSVSLSCRASQSISNNLHWYQQKSHESPRLLIKYVSQSSSGIPSRFSG"
    "SGSGTDFTLSINSVETEDFGMYFCQQSNSWPRTFGGGTKLEIK"
)
# Hen egg-white lysozyme from 1MLC chains E/F.
WILDTYPE_HEL = (
    "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWC"
    "NDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
)
CANONICAL = set(PROTEIN_ALPHABET)
MISSING_ABS = 1e-5
_SSML = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _col_index(ref: str) -> int:
    col = "".join(char for char in ref if char.isalpha())
    index = 0
    for char in col:
        index = index * 26 + ord(char) - 64
    return index - 1


def _read_xlsx_rows(path: Path) -> list[dict[int, str | None]]:
    with zipfile.ZipFile(path) as archive:
        ns = {"m": _SSML}
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", ns):
                shared.append(
                    "".join(
                        node.text or ""
                        for node in item.iter(f"{{{_SSML}}}t")
                    )
                )
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet.findall("m:sheetData/m:row", ns):
            values: dict[int, str | None] = {}
            for cell in row.findall("m:c", ns):
                value = cell.find("m:v", ns)
                if value is None:
                    continue
                text = (
                    shared[int(value.text)]
                    if cell.attrib.get("t") == "s"
                    else value.text
                )
                values[_col_index(cell.attrib["r"])] = text
            rows.append(values)
    return rows


def read_enrichment_table(path: Path) -> pd.DataFrame:
    """Parse S4 log-enrichment (VL then VH columns, amino-acid rows)."""
    rows = _read_xlsx_rows(path)
    if len(rows) < 4:
        raise ValueError(f"Unexpected enrichment workbook {path}.")
    pos_cols = sorted(col for col, value in rows[1].items() if value not in (None, ""))
    positions = [int(float(rows[1][col])) for col in pos_cols]
    wildtypes = [str(rows[2][col]) for col in pos_cols]
    split = next(
        i
        for i in range(1, len(positions))
        if positions[i] < positions[i - 1] and positions[i] <= 21
    )
    records = []
    for row in rows[3:]:
        mutant = row.get(0)
        if mutant not in CANONICAL:
            continue
        for index, (col, site, wildtype) in enumerate(
            zip(pos_cols, positions, wildtypes)
        ):
            raw = row.get(col)
            if raw in (None, ""):
                continue
            try:
                score = float(raw)
            except (TypeError, ValueError):
                continue
            if abs(score) < MISSING_ABS:
                continue
            chain = "VL" if index < split else "VH"
            records.append(
                {
                    "chain": chain,
                    "position": site,
                    "wildtype": wildtype,
                    "mutant": mutant,
                    "enrichment": score,
                }
            )
    return pd.DataFrame.from_records(records)


def _site_index(chain: str, site: int, vh: str = WILDTYPE_VH) -> int:
    if chain == "VH":
        return int(site) - 1
    if chain == "VL":
        return len(vh) + int(site) - 1
    raise ValueError(f"Unsupported chain {chain!r}.")


def _valid_mask(
    df: pd.DataFrame,
    vh: str = WILDTYPE_VH,
    vl: str = WILDTYPE_VL,
) -> pd.Series:
    sequences = {"VH": vh, "VL": vl}
    wt = df["wildtype"].astype(str)
    mut = df["mutant"].astype(str)
    ok = (
        df["chain"].isin(sequences)
        & (wt != mut)
        & wt.isin(CANONICAL)
        & mut.isin(CANONICAL)
        & df["enrichment"].notna()
    )
    letters = []
    for chain, site in zip(df["chain"], df["position"]):
        seq = sequences.get(str(chain), "")
        idx = int(site) - 1
        letters.append(seq[idx] if 0 <= idx < len(seq) else None)
    return ok & (wt.to_numpy() == np.asarray(letters, dtype=object))


def _sequence_record(vh: str, vl: str, hel: str) -> ak.Record:
    residue_label = ak.Array([list(vh), list(vl), list(hel)])
    residue_number = ak.Array(
        [
            np.arange(1, len(vh) + 1),
            np.arange(1, len(vl) + 1),
            np.arange(1, len(hel) + 1),
        ]
    )
    data = {
        "chain_label": ak.Array(["A", "B", "C"]),
        "residue_label": residue_label,
        "residue_number": residue_number,
    }
    data = {key: value[np.newaxis, np.newaxis] for key, value in data.items()}
    data["molecule_id"] = ak.Array([["D44_HEL"]])
    return ak.Record(data)


def _find_scores_xlsx(root) -> Path:
    matches = list(Path(root).rglob("*.xlsx"))
    if not matches:
        raise FileNotFoundError(f"No enrichment workbook found under {root}.")
    return matches[0]


def mutation_tables(
    df: pd.DataFrame,
    vh: str = WILDTYPE_VH,
    vl: str = WILDTYPE_VL,
):
    """Return concatenated Fv positions, mutant labels, and log-enrichment."""
    df = df[_valid_mask(df, vh, vl)].reset_index(drop=True)
    positions = [
        [_site_index(chain, site, vh)]
        for chain, site in zip(df["chain"], df["position"])
    ]
    labels = [[str(aa)] for aa in df["mutant"]]
    effects = [float(value) for value in df["enrichment"]]
    return positions, labels, effects


class WarszawskiD44Adapter(Adapter):
    """D44.1 anti-lysozyme Fv deep mutational scan from Warszawski et al."""

    @classmethod
    def download(
        cls,
        model: str = "boltz2",
        use_msa_server: bool = True,
        cache_dir: str | Path | None = None,
    ):
        path = config.raw_path / "Warszawski_D44"
        dest = path / "S4_d44_enrichment.xlsx"
        download(SCORES_URL, dest, extension=".xlsx")
        df = read_enrichment_table(_find_scores_xlsx(path))
        positions, labels, effects = mutation_tables(df)

        record = _sequence_record(WILDTYPE_VH, WILDTYPE_VL, WILDTYPE_HEL)
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
            {"warszawski_mutation_split": ["test"] * len(effects)},
            default="warszawski_mutation_split",
        )
        assets = Assets(
            {
                "vh_sequence": WILDTYPE_VH,
                "vl_sequence": WILDTYPE_VL,
                "hel_sequence": WILDTYPE_HEL,
            }
        )
        transform = StructurePredictionTransform(
            model=model,
            use_msa_server=use_msa_server,
            cache_dir=cache_dir,
        )
        return transform(batches, split, assets)
