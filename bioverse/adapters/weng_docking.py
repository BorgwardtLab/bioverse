import tarfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import awkward as ak
import numpy as np
from biopandas.pdb import PandasPdb
from biopandas.pdb.engines import amino3to1dict

from ..adapter import Adapter
from ..data import Assets, Split
from ..processors import PdbProcessor
from ..utilities import IteratorWithLength, batched, config, download

BASE_URL = "https://zlab.wenglab.org/benchmark"
STRUCTURE_PARTS = ("r_b", "l_b", "r_u", "l_u")
MOLECULE_FIELDS = (
    "residue_number",
    "residue_label",
    "atom_label",
    "atom_pos",
    "atom_b_factor",
)
METADATA_FIELDS = (
    "complex",
    "category",
    "difficulty",
    "irmsd",
    "dasa",
    "pdb1",
    "pdb2",
)


def parse_benchmark_table(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as archive:
        shared_strings = []
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        for item in root.findall(".//m:si", namespace):
            texts = [node.text or "" for node in item.findall(".//m:t", namespace)]
            shared_strings.append("".join(texts))

        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in root.findall(".//m:row", namespace):
            values = []
            for cell in row.findall("m:c", namespace):
                value = cell.find("m:v", namespace)
                if value is None:
                    values.append("")
                elif cell.get("t") == "s":
                    values.append(shared_strings[int(value.text)])
                else:
                    values.append(value.text)
            rows.append(values)

    difficulty = None
    cases = []
    for row in rows:
        if row and row[0]:
            if "Rigid-body" in row[0]:
                difficulty = "rigid"
            elif "Medium Difficulty" in row[0]:
                difficulty = "medium"
            elif row[0].startswith("Difficult"):
                difficulty = "difficult"
        if len(row) < 9 or not row[0] or not row[0][0].isdigit() or ":" not in row[0]:
            continue
        cases.append(
            {
                "complex": row[0],
                "code": row[0].split("_", 1)[0],
                "category": row[1],
                "pdb1": row[2],
                "protein1": row[3],
                "pdb2": row[4],
                "protein2": row[5],
                "irmsd": float(row[6]),
                "dasa": float(row[7]),
                "version": row[8],
                "difficulty": difficulty,
            }
        )
    return cases


def _flat_chains(record, field: str) -> ak.Array:
    values = record[field][0][0]
    if field == "chain_label":
        counts = ak.to_numpy(ak.num(record.residue_label[0][0], axis=-1))
        return ak.Array(np.repeat(ak.to_numpy(values), counts))
    return ak.concatenate(values, axis=0)


def merge_structures(records, molecule_id: str, metadata: dict) -> ak.Record:
    flat = {
        field: ak.concatenate([_flat_chains(record, field) for record in records])
        for field in ("chain_label", *MOLECULE_FIELDS)
        if field == "chain_label" or field in records[0].fields
    }
    merged = {
        field: ak.Array([[flat[field]]])
        for field in flat
    }
    merged["molecule_id"] = ak.Array([[molecule_id]])
    for key, value in metadata.items():
        merged[f"molecule_{key}"] = ak.Array([[value]])
    return ak.Record({key: value[np.newaxis] for key, value in merged.items()})


class WengPdbProcessor(PdbProcessor):
    """Parse Weng docking benchmark PDB files."""

    @classmethod
    def process_file(cls, path: str | Path, pLDDT: bool = False) -> ak.Record:
        path = Path(path)
        pdb = PandasPdb().read_pdb(str(path))
        is_alphafold = (
            pdb.df["OTHERS"]["entry"].str.contains("ALPHAFOLD", case=False).any()
        )
        name = path.name.split(".")[0]
        df = pdb.df["ATOM"]
        chain_sizes = df.groupby("chain_id", sort=False).size().to_numpy()
        residue_sizes = (
            df.groupby(["chain_id", "residue_number"], sort=False).size().to_numpy()
        )
        data = {
            "chain_label": df["chain_id"].to_list(),
            "residue_number": df["residue_number"].to_numpy(),
            "residue_label": df["residue_name"].map(amino3to1dict).to_list(),
            "atom_label": df["atom_name"].to_list(),
            "atom_pos": df[["x_coord", "y_coord", "z_coord"]].to_numpy(),
            "atom_b_factor": df["b_factor"].to_numpy(),
        }
        data = {
            key: ak.unflatten(
                ak.unflatten(ak.Array(value), chain_sizes), residue_sizes, axis=1
            )
            for key, value in data.items()
        }
        data["residue_number"] = data["residue_number"].firsts(2)
        data["residue_label"] = data["residue_label"].firsts(2)
        data["chain_label"] = data["chain_label"].firsts(2).firsts(1)
        if is_alphafold or pLDDT:
            data["residue_pLDDT"] = data.pop("atom_b_factor").firsts(2)
        data = {key: value[np.newaxis, np.newaxis] for key, value in data.items()}
        data["molecule_id"] = ak.Array([[name]])
        return ak.Record(data)


class WengDockingAdapter(Adapter):
    """Download protein-ligand complexes from the Weng et al. docking benchmark."""

    @classmethod
    def download(cls, version: str = "5.5"):
        path = config.raw_path / "WengDocking" / version
        archive = path / f"benchmark{version}.tgz"
        benchmark_dir = path / f"benchmark{version}"
        table_path = path / f"Table_BM{version}.xlsx"
        struct_dir = benchmark_dir / "structures"

        if not struct_dir.exists():
            download(f"{BASE_URL}/benchmark{version}.tgz", path / f"benchmark{version}")
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(path)

        if not table_path.exists():
            download(f"{BASE_URL}/Table_BM{version}.xlsx", table_path.with_suffix(""))

        cases = parse_benchmark_table(table_path)
        records = []
        complex_ids = []
        unbound_paths = {}
        for case in cases:
            paths = {
                part: struct_dir / f"{case['code']}_{part}.pdb" for part in STRUCTURE_PARTS
            }
            if not all(path.exists() for path in paths.values()):
                continue
            receptor_bound, ligand_bound = [
                WengPdbProcessor.process_file(paths[part]) for part in ("r_b", "l_b")
            ]
            metadata = {key: case[key] for key in METADATA_FIELDS}
            records.append(
                merge_structures(
                    [receptor_bound, ligand_bound], case["complex"], metadata
                )
            )
            complex_ids.append(case["complex"])
            unbound_paths[case["complex"]] = {
                part: str(paths[part]) for part in ("r_u", "l_u")
            }

        assets = Assets(
            {
                "complex_ids": complex_ids,
                "unbound_structures": unbound_paths,
            }
        )
        return (
            batched(IteratorWithLength(iter(records), len(records))),
            Split(),
            assets,
        )
