"""Adapter for the BioLiP protein–ligand set with Kd annotations."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb

from ..adapter import Adapter
from ..data import Assets, Split
from ..utilities import (
    CHAIN_ALPHABET,
    THREE_TO_ONE,
    IteratorWithLength,
    batched,
    config,
    download,
    element_from_atom_name,
    note,
    normalize_element,
    parallelize,
    progressbar,
)

ANNOTATION_URLS = (
    "https://zhanggroup.org/BioLiP/download/BioLiP.txt.gz",
    "https://seq2fun.dcmb.med.umich.edu/BioLiP2/download/BioLiP.txt.gz",
)
POLYMER_LIGANDS = frozenset({"dna", "rna", "peptide"})
AFFINITY_SOURCES = ("manual", "moad", "pdbbind", "bindingdb")
EMPTY_AFFINITY = frozenset(
    {"", ".", "-", "none", "null", "nan", "remove_due_to_license_issue"}
)
KD_RE = re.compile(
    r"(?i)\bkd\s*=\s*([<>~])?\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)\s*([a-zA-Zµμu]+)?"
)
SITE_RE = re.compile(r"([A-Z])(-?\d+)([A-Z])?")
R_KCAL = 1.98720425864083e-3  # kcal / (mol K)
T_REF = 298.15  # K; ΔG = RT ln(Kd)
UNIT_TO_M = {
    "fm": 1e-15,
    "pm": 1e-12,
    "nm": 1e-9,
    "um": 1e-6,
    "µm": 1e-6,
    "μm": 1e-6,
    "mm": 1e-3,
    "m": 1.0,
}


def is_polymer_ligand(ligand_id: str) -> bool:
    return ligand_id.strip().lower() in POLYMER_LIGANDS


def is_peptide_ligand(ligand_id: str) -> bool:
    return ligand_id.strip().lower() == "peptide"


def parse_kd(text: str):
    """Return Kd in molar units, or ``None`` if missing, bounded, or not Kd."""
    if text is None:
        return None
    text = str(text).strip()
    if text.lower() in EMPTY_AFFINITY:
        return None
    match = KD_RE.search(text)
    if match is None:
        return None
    bound, value, unit = match.group(1), match.group(2), match.group(3)
    if bound in {"<", ">"}:
        return None
    try:
        kd = float(value)
    except ValueError:
        return None
    if not math.isfinite(kd) or kd <= 0:
        return None
    scale = UNIT_TO_M.get((unit or "M").lower(), None)
    if scale is None:
        return None
    return kd * scale


def parse_affinity_fields(fields: list[str]):
    """Return the first valid Kd across BioLiP affinity columns.

    Column order is manual literature, Binding MOAD, PDBbind-CN, BindingDB.
    """
    for source, field in zip(AFFINITY_SOURCES, fields):
        kd = parse_kd(field)
        if kd is None:
            continue
        return {
            "kd": kd,
            "pkd": -math.log10(kd),
            "dg": R_KCAL * T_REF * math.log(kd),
            "affinity_source": source,
        }
    return None


def parse_binding_site(text: str) -> list[tuple[str, int, str]]:
    """Parse BioLiP binding-site tokens ``X123`` / ``F100A`` into contacts."""
    contacts = []
    if not text or not str(text).strip():
        return contacts
    for token in str(text).split():
        match = SITE_RE.fullmatch(token.strip())
        if match is None:
            continue
        contacts.append((match.group(1), int(match.group(2)), match.group(3) or ""))
    return contacts


def parse_annotation_line(line: str, ligand: str = "small_molecule") -> dict | None:
    """Parse one BioLiP annotation row.

    ``ligand="small_molecule"`` keeps Kd-annotated organic ligands and drops
    polymers. ``ligand="peptide"`` keeps peptide complexes and stores no
    affinity labels.
    """
    if ligand not in {"small_molecule", "peptide"}:
        raise ValueError(f"Unknown BioLiP ligand kind {ligand!r}.")
    cols = line.rstrip("\n").split("\t")
    if len(cols) < 21:
        return None
    ligand_id = cols[4].strip()
    if ligand == "peptide":
        if not is_peptide_ligand(ligand_id):
            return None
        affinity = None
    else:
        if is_polymer_ligand(ligand_id):
            return None
        affinity = parse_affinity_fields(cols[13:17])
        if affinity is None:
            return None
    try:
        ligand_resi = int(str(cols[19]).strip())
    except ValueError:
        ligand_resi = None
    try:
        serial = int(str(cols[6]).strip())
    except ValueError:
        serial = 1
    try:
        resolution = float(cols[2])
    except ValueError:
        resolution = -1.0
    pdb_id = cols[0].strip().lower()
    receptor_chain = cols[1].strip()
    ligand_chain = cols[5].strip()
    entry = {
        "pdb_id": pdb_id,
        "receptor_chain": receptor_chain,
        "ligand_id": ligand_id,
        "ligand_chain": ligand_chain,
        "ligand_serial": serial,
        "ligand_resi": ligand_resi,
        "resolution": resolution,
        "binding_site": parse_binding_site(cols[7]),
        "uniprot_id": cols[17].strip(),
        "molecule_id": (
            f"{pdb_id}_{receptor_chain}_{ligand_id}_{ligand_chain}_{serial}"
        ),
    }
    if affinity is not None:
        entry.update(affinity)
    return entry


def parse_annotations(path: Path | str, ligand: str = "small_molecule") -> list[dict]:
    path = Path(path)
    entries = []
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            entry = parse_annotation_line(line, ligand=ligand)
            if entry is not None:
                entries.append(entry)
    return entries


def _insertion(value) -> str:
    text = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
    text = text.strip()
    return "" if text in {".", "?", "nan"} else text


_ATOM_COLUMNS = (
    "chain_id",
    "residue_number",
    "residue_name",
    "insertion",
    "atom_name",
    "element",
    "x",
    "y",
    "z",
    "b_factor",
    "group",
)


def _elements_from_frame(names, symbols=None):
    if symbols is None:
        return [element_from_atom_name(name) for name in names]
    return [
        normalize_element(symbol)
        if pd.notna(symbol) and str(symbol).strip()
        else element_from_atom_name(name)
        for symbol, name in zip(symbols, names)
    ]


def _empty_atoms() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_ATOM_COLUMNS))


def _pdb_atoms(path: Path) -> pd.DataFrame:
    pdb = PandasPdb().read_pdb(str(path))
    frames = []
    for group in ("ATOM", "HETATM"):
        df = pdb.df.get(group)
        if df is None or df.empty:
            continue
        frame = pd.DataFrame(
            {
                "chain_id": df["chain_id"].astype(str),
                "residue_number": pd.to_numeric(df["residue_number"], errors="coerce"),
                "residue_name": df["residue_name"].astype(str).str.strip(),
                "insertion": (
                    df["insertion"].map(_insertion)
                    if "insertion" in df
                    else pd.Series([""] * len(df), index=df.index)
                ),
                "atom_name": df["atom_name"].astype(str).str.strip(),
                "element": _elements_from_frame(
                    df["atom_name"].astype(str).str.strip(),
                    df["element_symbol"] if "element_symbol" in df else None,
                ),
                "x": df["x_coord"].astype(float),
                "y": df["y_coord"].astype(float),
                "z": df["z_coord"].astype(float),
                "b_factor": df["b_factor"].astype(float),
                "group": group,
            }
        )
        frames.append(frame)
    if not frames:
        return _empty_atoms()
    atoms = pd.concat(frames, ignore_index=True)
    atoms = atoms.dropna(subset=["residue_number"])
    atoms["residue_number"] = atoms["residue_number"].astype(int)
    return atoms.drop_duplicates(
        ["chain_id", "residue_number", "insertion", "residue_name", "atom_name"],
        keep="first",
    )


def _cif_atoms(path: Path) -> pd.DataFrame:
    from biopandas.mmcif import PandasMmcif

    mmcif = PandasMmcif().read_mmcif(str(path))
    frames = []
    for key in ("ATOM", "HETATM"):
        df = mmcif.df.get(key)
        if df is None or df.empty:
            continue
        chain = df["auth_asym_id"] if "auth_asym_id" in df else df["label_asym_id"]
        resi = df["auth_seq_id"] if "auth_seq_id" in df else df["label_seq_id"]
        resname = df["auth_comp_id"] if "auth_comp_id" in df else df["label_comp_id"]
        atom = df["auth_atom_id"] if "auth_atom_id" in df else df["label_atom_id"]
        insertion = (
            df["pdbx_PDB_ins_code"]
            if "pdbx_PDB_ins_code" in df
            else pd.Series([""] * len(df), index=df.index)
        )
        names = atom.astype(str).str.strip()
        symbols = df["type_symbol"] if "type_symbol" in df else None
        frames.append(
            pd.DataFrame(
                {
                    "chain_id": chain.astype(str),
                    "residue_number": pd.to_numeric(resi, errors="coerce"),
                    "residue_name": resname.astype(str).str.strip(),
                    "insertion": insertion.map(_insertion),
                    "atom_name": names,
                    "element": _elements_from_frame(names, symbols),
                    "x": df["Cartn_x"].astype(float),
                    "y": df["Cartn_y"].astype(float),
                    "z": df["Cartn_z"].astype(float),
                    "b_factor": pd.to_numeric(
                        df.get("B_iso_or_equiv", 0.0), errors="coerce"
                    ).fillna(0.0),
                    "group": df["group_PDB"].astype(str)
                    if "group_PDB" in df
                    else key,
                }
            )
        )
    if not frames:
        return _empty_atoms()
    atoms = pd.concat(frames, ignore_index=True)
    atoms = atoms.dropna(subset=["residue_number"])
    atoms["residue_number"] = atoms["residue_number"].astype(int)
    return atoms.drop_duplicates(
        ["chain_id", "residue_number", "insertion", "residue_name", "atom_name"],
        keep="first",
    )


def load_structure_atoms(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.startswith(".cif"):
        return _cif_atoms(path)
    return _pdb_atoms(path)


def _group_residues(df: pd.DataFrame, amino_acid_labels: bool) -> list[dict]:
    residues = []
    grouped = df.groupby(
        ["chain_id", "residue_number", "insertion", "residue_name"], sort=False
    )
    for (chain, number, insertion, name), group in grouped:
        label = THREE_TO_ONE.get(name, name) if amino_acid_labels else name
        names = group["atom_name"].tolist()
        residues.append(
            {
                "chain": str(chain),
                "number": int(number),
                "insertion": _insertion(insertion),
                "label": str(label),
                "atom_name": names,
                "atom_label": names
                if amino_acid_labels
                else group["element"].tolist(),
                "atom_pos": group[["x", "y", "z"]].to_numpy(dtype=np.float32),
                "atom_b_factor": group["b_factor"].to_numpy(dtype=np.float32),
            }
        )
    return residues


def _select_ligand(df: pd.DataFrame, entry: dict) -> pd.DataFrame:
    hits = df[
        (df["residue_name"] == entry["ligand_id"])
        & (df["chain_id"] == entry["ligand_chain"])
    ]
    if hits.empty:
        return hits
    if entry["ligand_resi"] is not None:
        by_resi = hits[hits["residue_number"] == entry["ligand_resi"]]
        if not by_resi.empty:
            return by_resi
    residue_ids = list(
        hits.groupby(["residue_number", "insertion"], sort=False).groups
    )
    index = max(entry["ligand_serial"] - 1, 0)
    if index >= len(residue_ids):
        return hits.iloc[0:0]
    number, insertion = residue_ids[index]
    return hits[
        (hits["residue_number"] == number) & (hits["insertion"] == insertion)
    ]


def _ligand_chain_label(protein_chain: str, ligand_chain: str) -> str:
    if ligand_chain and ligand_chain != protein_chain:
        return ligand_chain
    for letter in CHAIN_ALPHABET:
        if letter != protein_chain:
            return letter
    return "L"


def _contact_mask(residues: list[dict], contacts: list[tuple[str, int, str]]) -> list[int]:
    wanted = set(contacts)
    return [
        int((res["label"], res["number"], res["insertion"]) in wanted)
        for res in residues
    ]


def _chain_arrays(
    residues: list[dict],
    chain_label: str,
    chain_type: str,
    contacts: list[int] | None = None,
):
    data = {
        "chain_label": chain_label,
        "chain_type": chain_type,
        "residue_label": [res["label"] for res in residues],
        "residue_number": [res["number"] for res in residues],
        "residue_insertion": [res["insertion"] for res in residues],
        "atom_name": [res["atom_name"] for res in residues],
        "atom_label": [res["atom_label"] for res in residues],
        "atom_pos": [res["atom_pos"] for res in residues],
        "atom_b_factor": [res["atom_b_factor"] for res in residues],
    }
    if contacts is not None:
        data["residue_contact"] = contacts
    return data


def build_complex_record(atoms: pd.DataFrame, entry: dict) -> ak.Record | None:
    """Build one molecule with a protein chain and a ligand chain."""
    protein_df = atoms[
        (atoms["group"] == "ATOM") & (atoms["chain_id"] == entry["receptor_chain"])
    ]
    ligand_df = _select_ligand(atoms, entry)
    if protein_df.empty or ligand_df.empty:
        return None

    protein = _group_residues(protein_df, amino_acid_labels=True)
    ligand = _group_residues(ligand_df, amino_acid_labels=False)
    if not protein or not ligand:
        return None

    ligand_chain = _ligand_chain_label(entry["receptor_chain"], entry["ligand_chain"])
    protein_chain = _chain_arrays(
        protein,
        entry["receptor_chain"],
        "protein",
        _contact_mask(protein, entry["binding_site"]),
    )
    ligand_chain_data = _chain_arrays(
        ligand,
        ligand_chain,
        "ligand",
        [0] * len(ligand),
    )

    data = {}
    for key in protein_chain:
        data[key] = ak.Array([protein_chain[key], ligand_chain_data[key]])
    data = {key: value[np.newaxis, np.newaxis] for key, value in data.items()}
    data["molecule_id"] = ak.Array([[entry["molecule_id"]]])
    for key in (
        "kd",
        "pkd",
        "dg",
        "pdb_id",
        "ligand_id",
        "receptor_chain",
        "ligand_chain",
        "affinity_source",
        "resolution",
        "uniprot_id",
    ):
        data[f"molecule_{key}"] = ak.Array([[entry[key]]])
    return ak.Record(data)


def _standard_amino_acids(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["residue_name"].isin(THREE_TO_ONE)]


def _select_peptide(df: pd.DataFrame, entry: dict) -> pd.DataFrame:
    hits = df[df["chain_id"] == entry["ligand_chain"]]
    if hits.empty:
        return hits
    return _standard_amino_acids(hits)


def build_peptide_complex_record(
    atoms: pd.DataFrame, entry: dict, max_length: int | None = None
) -> ak.Record | None:
    """Build one unlabeled protein–peptide molecule as two protein-like chains."""
    if entry["receptor_chain"] == entry["ligand_chain"]:
        return None
    protein_df = _standard_amino_acids(
        atoms[
            (atoms["group"] == "ATOM") & (atoms["chain_id"] == entry["receptor_chain"])
        ]
    )
    peptide_df = _select_peptide(atoms, entry)
    if protein_df.empty or peptide_df.empty:
        return None

    protein = _group_residues(protein_df, amino_acid_labels=True)
    peptide = _group_residues(peptide_df, amino_acid_labels=True)
    if not protein or not peptide:
        return None
    if max_length is not None and (
        len(protein) >= max_length or len(peptide) >= max_length
    ):
        return None

    protein_chain = _chain_arrays(protein, entry["receptor_chain"], "protein")
    peptide_chain = _chain_arrays(peptide, entry["ligand_chain"], "peptide")

    data = {}
    for key in protein_chain:
        data[key] = ak.Array([protein_chain[key], peptide_chain[key]])
    data = {key: value[np.newaxis, np.newaxis] for key, value in data.items()}
    data["molecule_id"] = ak.Array([[entry["molecule_id"]]])
    for key in (
        "pdb_id",
        "ligand_id",
        "receptor_chain",
        "ligand_chain",
        "resolution",
        "uniprot_id",
    ):
        data[f"molecule_{key}"] = ak.Array([[entry[key]]])
    return ak.Record(data)


def structure_urls(pdb_id: str) -> list[str]:
    pdb_id = pdb_id.lower()
    return [
        f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb",
        f"https://files.rcsb.org/download/{pdb_id.upper()}.cif",
        f"https://www.ebi.ac.uk/pdbe/entry-files/download/{pdb_id}.cif",
    ]


def _existing_structure(directory: Path, pdb_id: str) -> Path | None:
    for suffix in (".pdb", ".cif", ".pdb.gz", ".cif.gz"):
        path = directory / f"{pdb_id}{suffix}"
        if path.exists():
            return path
    return None


def _resolved_download(
    url: str, dest_stem: Path, progress: bool = True
) -> Path | None:
    dest_stem = Path(dest_stem)
    suffix = "".join(Path(url.split("?")[0]).suffixes)
    expected = [
        dest_stem,
        dest_stem.with_suffix(suffix),
        dest_stem.parent / f"{dest_stem.name}.txt",
    ]
    for path in expected:
        if path.exists():
            return path
    try:
        download(url, dest_stem, progress=progress)
    except Exception as exc:
        if progress:
            note(f"Download failed for {url}: {exc}")
        return None
    for path in expected:
        if path.exists():
            return path
    matches = sorted(dest_stem.parent.glob(f"{dest_stem.name}*"))
    return next((path for path in matches if path.is_file()), None)


def ensure_structure(
    pdb_id: str, directory: Path, progress: bool = True
) -> Path | None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    existing = _existing_structure(directory, pdb_id)
    if existing is not None:
        return existing
    for url in structure_urls(pdb_id):
        path = _resolved_download(url, directory / pdb_id, progress=progress)
        if path is not None:
            return path
    return None


def ensure_annotations(directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    unzipped = directory / "BioLiP.txt"
    if unzipped.exists():
        return unzipped
    for url in ANNOTATION_URLS:
        path = _resolved_download(url, directory / "BioLiP")
        if unzipped.exists():
            return unzipped
        if path is not None and path.exists() and path.suffix == ".txt":
            return path
    raise FileNotFoundError(f"BioLiP annotation file not found under {directory}.")


def _thread_count(workers: int | None = None, cap: int | None = None) -> int:
    n = int(workers if workers is not None else config.workers)
    n = max(1, n)
    return min(n, cap) if cap is not None else n


def _imap_threads(fn, items, description: str, workers: int):
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [fn(item) for item in progressbar(items, description=description)]
    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for future in progressbar(
            as_completed(futures),
            total=len(futures),
            description=description,
        ):
            results[futures[future]] = future.result()
    return results


def _download_one(item: tuple[str, Path]) -> tuple[str, Path | None]:
    pdb_id, directory = item
    return pdb_id, ensure_structure(pdb_id, directory, progress=False)


def _download_structures(
    pdb_ids: list[str], directory: Path, workers: int | None = None
) -> dict[str, Path | None]:
    directory.mkdir(parents=True, exist_ok=True)
    workers = _thread_count(workers, cap=32)
    pairs = _imap_threads(
        _download_one,
        [(pdb_id, directory) for pdb_id in pdb_ids],
        "Downloading structures",
        workers,
    )
    return dict(pairs)


def _build_pdb_complexes(
    item: tuple[Path, list[dict], str, int | None],
) -> list[ak.Record | None]:
    path, entries, ligand, max_length = item
    try:
        atoms = load_structure_atoms(path)
    except Exception:
        return [None] * len(entries)
    records = []
    for entry in entries:
        try:
            if ligand == "peptide":
                record = build_peptide_complex_record(
                    atoms, entry, max_length=max_length
                )
            else:
                record = build_complex_record(atoms, entry)
        except Exception:
            record = None
        records.append(record)
    return records


def _build_complexes(
    entries: list[dict],
    seen_pdb: dict[str, Path | None],
    ligand: str,
    workers: int | None = None,
    max_length: int | None = None,
) -> tuple[list[ak.Record], dict[str, int]]:
    by_pdb: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        by_pdb[entry["pdb_id"]].append(entry)
    skipped = {"missing_structure": 0, "missing_atoms": 0}
    tasks = []
    for pdb_id, group in by_pdb.items():
        path = seen_pdb.get(pdb_id)
        if path is None:
            skipped["missing_structure"] += len(group)
            continue
        tasks.append((path, group, ligand, max_length))
    records = []
    for chunk in parallelize(
        _build_pdb_complexes,
        tasks,
        description="Building complexes",
        max_workers=workers,
    ):
        for record in chunk:
            if record is None:
                skipped["missing_atoms"] += 1
            else:
                records.append(record)
    return records, skipped


class BioLipAdapter(Adapter):
    """Download BioLiP complexes.

    ``ligand="small_molecule"`` (default) keeps protein–ligand pairs with
    experimental Kd. ``ligand="peptide"`` keeps unlabeled protein–peptide
    complexes for masked residue prediction.
    """

    @classmethod
    def download(
        cls,
        limit: int | None = None,
        ligand: str = "small_molecule",
        max_length: int | None = None,
    ):
        if ligand not in {"small_molecule", "peptide"}:
            raise ValueError(f"Unknown BioLiP ligand kind {ligand!r}.")
        root = config.raw_path / "BioLiP"
        annotations = ensure_annotations(root)
        entries = parse_annotations(annotations, ligand=ligand)
        if limit is not None:
            entries = entries[:limit]

        structures = root / "structures"
        pdb_ids = list(dict.fromkeys(entry["pdb_id"] for entry in entries))
        seen_pdb = _download_structures(pdb_ids, structures)
        records, skipped = _build_complexes(
            entries, seen_pdb, ligand, max_length=max_length
        )

        kind = "peptide" if ligand == "peptide" else "Kd"
        note(
            f"BioLiP {kind} complexes: {len(records)} kept, "
            f"{skipped['missing_structure']} missing structures, "
            f"{skipped['missing_atoms']} missing atoms."
        )
        assets = Assets(
            {
                "n_annotations": len(entries),
                "n_complexes": len(records),
                "ligand": ligand,
                "max_length": max_length,
                "skipped": skipped,
            }
        )
        return (
            batched(IteratorWithLength(iter(records), len(records))),
            Split(),
            assets,
        )
