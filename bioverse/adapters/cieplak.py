"""Cieplak et al. quantitative MMP-2 / MMP-9 substrate specificity."""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import awkward as ak
import numpy as np

from ..adapter import Adapter
from ..data import Assets, Split
from ..utilities import PROTEIN_ALPHABET, config, download
from ..utilities.cleavage import WINDOW_LEN, pad_window, residue_letter
from .merops import _attach_pairs
from .protease_structures import records_to_batches, resolve_protease_structure

SCORES_URL = (
    "https://journals.plos.org/ploscompbiol/article/file"
    "?id=10.1371/journal.pcbi.1008101.s010&type=supplementary"
)
# MMP-2 gelatinase A; MMP-9 gelatinase B.
PROTEASES = {
    "MMP2": {"uniprot": "P08253", "merops": "M10.003"},
    "MMP9": {"uniprot": "P14780", "merops": "M10.004"},
}
_SSML = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
CANONICAL = set(PROTEIN_ALPHABET)


def _col_index(ref: str) -> int:
    col = "".join(char for char in ref if char.isalpha())
    index = 0
    for char in col:
        index = index * 26 + ord(char) - 64
    return index - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    ns = {"m": _SSML}
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    shared = []
    for item in root.findall("m:si", ns):
        shared.append(
            "".join(node.text or "" for node in item.iter(f"{{{_SSML}}}t"))
        )
    return shared


def _sheet_rows(archive: zipfile.ZipFile, sheet: str, shared: list[str]):
    ns = {"m": _SSML}
    root = ET.fromstring(archive.read(sheet))
    rows = []
    for row in root.findall("m:sheetData/m:row", ns):
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
        if values:
            rows.append(values)
    return rows


def _looks_like_peptide(text: str) -> bool:
    letters = [residue_letter(ch) for ch in str(text).strip()]
    n_aa = sum(ch in CANONICAL for ch in letters)
    return 4 <= len(letters) <= 12 and n_aa >= 4 and all(
        ch in CANONICAL or ch == "-" for ch in letters
    )


def parse_cieplak_xlsx(path: Path) -> dict[str, list[tuple[str, float]]]:
    """Return {MMP2|MMP9: [(window, kcat/KM), ...]} from the S10 workbook."""
    path = Path(path)
    out: dict[str, list[tuple[str, float]]] = {"MMP2": [], "MMP9": []}
    with zipfile.ZipFile(path) as archive:
        shared = _shared_strings(archive)
        sheets = [
            name
            for name in archive.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        ]
        for i, sheet in enumerate(sorted(sheets)):
            key = "MMP2" if i == 0 else "MMP9" if i == 1 else None
            if key is None:
                continue
            rows = _sheet_rows(archive, sheet, shared)
            if not rows:
                continue
            header = rows[0]
            seq_col = None
            val_col = None
            for idx, value in header.items():
                name = str(value or "").lower()
                if seq_col is None and any(
                    token in name for token in ("hexamer", "sequence", "substrate")
                ):
                    seq_col = idx
                if val_col is None and "kcat" in name:
                    val_col = idx
            if seq_col is None:
                seq_col = 1 if 1 in header else 0
            if val_col is None:
                val_col = 5 if 5 in header else (max(header) if header else 4)
            for row in rows[1:]:
                seq = row.get(seq_col)
                raw = row.get(val_col)
                if seq is None or raw is None:
                    continue
                if not _looks_like_peptide(seq):
                    continue
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                window = pad_window(
                    "".join(residue_letter(ch) for ch in str(seq).strip()),
                    align="p4",
                )
                out[key].append((window, value))
    return out


class CieplakAdapter(Adapter):
    """Quantitative MMP-2 / MMP-9 hexamer kcat/KM measurements from Cieplak et al."""

    @classmethod
    def download(cls, cache_dir: str | Path | None = None):
        root = Path(cache_dir or (config.raw_path / "CIEPLK"))
        dest = root / "cieplak_s10.xlsx"
        download(SCORES_URL, dest, extension=".xlsx")
        matches = list(root.rglob("*.xlsx"))
        if not matches:
            raise FileNotFoundError(f"Cieplak workbook not found under {root}.")
        pairs = parse_cieplak_xlsx(matches[0])
        pdb_dir = root / "pdb"
        af_dir = root / "alphafold"
        mapping_dir = root / "pdbe"
        records = []
        for name, meta in PROTEASES.items():
            resolved = resolve_protease_structure(
                meta["uniprot"],
                pdb_dir=pdb_dir,
                af_dir=af_dir,
                mapping_dir=mapping_dir,
                molecule_id=name,
            )
            if resolved is None:
                raise RuntimeError(f"No structure for {name} ({meta['uniprot']}).")
            record, source = resolved
            if not pairs[name]:
                raise RuntimeError(f"No Cieplak peptides parsed for {name}.")
            windows, effects = zip(*pairs[name])
            record = _attach_pairs(record, list(windows), list(effects))
            data = {field: record[field] for field in record.fields}
            data["molecule_uniprot"] = ak.Array([[meta["uniprot"]]])
            data["molecule_merops_id"] = ak.Array([[meta["merops"]]])
            data["molecule_structure_source"] = ak.Array([[source]])
            records.append(ak.Record(data))
        batches = records_to_batches(records)
        split = Split(
            {"cieplak_scene_split": ["test"] * len(records)},
            default="cieplak_scene_split",
        )
        assets = Assets(
            {
                "window_len": WINDOW_LEN,
                "proteases": PROTEASES,
            }
        )
        return batches, split, assets
