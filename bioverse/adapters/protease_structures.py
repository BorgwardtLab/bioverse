"""Download protease structures from PDB when possible, otherwise AlphaFold DB."""

from __future__ import annotations

from pathlib import Path

import awkward as ak
import numpy as np
import requests

from ..processors.cif import CifProcessor
from ..processors.pdb import PdbProcessor
from ..utilities import load, note, save
from ..utilities.cleavage import CANONICAL, residue_letter
from ..utilities.id_mapping import fetch_alphafold_structure
from .biolip import ensure_structure

PDBE_BEST = "https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/{uniprot}"


def _peel_chains(value):
    value = ak.to_list(value)
    while (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], list)
        and value[0]
        and not isinstance(value[0][0], str)
    ):
        value = value[0]
    return value


def _chain_count(residue_label) -> int:
    chains = _peel_chains(residue_label)
    if not chains:
        return 0
    if isinstance(chains[0], str):
        return 1
    return len(chains)


def _peel_to_chain_axis(value):
    peeled = value
    while getattr(peeled, "ndim", 0) > 1 and len(peeled) == 1:
        peeled = peeled[0]
    return peeled


def scene_frame(value) -> ak.Array:
    """Wrap chain-level data as scene / frame / single-chain."""
    if isinstance(value, (str, bytes, np.str_)):
        payload = str(value)
    elif isinstance(value, ak.Array):
        payload = ak.to_list(value)
    elif isinstance(value, np.ndarray):
        payload = value.tolist()
    else:
        payload = value
    return ak.Array([[[payload]]])


def keep_longest_chain(record: ak.Record) -> ak.Record:
    """Keep the polymer chain with the most residues."""
    n_chains = _chain_count(record["residue_label"])
    if n_chains <= 1:
        return record
    labels = _peel_chains(record["residue_label"])
    lengths = [len(chain) if chain is not None else 0 for chain in labels]
    best = int(np.argmax(lengths))
    out = {}
    for field in record.fields:
        value = record[field]
        if field.startswith("molecule_"):
            out[field] = value
            continue
        peeled = _peel_to_chain_axis(value)
        try:
            n_items = len(peeled)
        except TypeError:
            out[field] = value
            continue
        if n_items != n_chains:
            out[field] = value
            continue
        out[field] = scene_frame(peeled[best])
    if "molecule_id" not in out:
        out["molecule_id"] = ak.Array([["protease"]])
    return ak.Record(out)


def _canonical_letter(token) -> str:
    letter = residue_letter(token)
    return letter if letter in CANONICAL else "X"


def _to_python(value):
    return ak.to_list(value) if isinstance(value, (ak.Array, ak.Record)) else value


def _is_xyz(value) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(item, (int, float, np.floating)) for item in value)
    )


def _longest_or_unwrap(value):
    while isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) == 1:
            value = value[0]
            continue
        value = max(value, key=lambda item: len(item) if item is not None else 0)
    return value


def residue_letters_from(value) -> list[str]:
    value = _to_python(value)
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, str):
        return [_canonical_letter(ch) for ch in value]
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = max(value, key=lambda chain: len(chain) if chain else 0)
        if isinstance(value, list) and value and isinstance(value[0], list):
            value = max(value, key=len)
    if isinstance(value, list):
        return [_canonical_letter(item) for item in value]
    return [_canonical_letter(value)]


def _per_residue_lists(value, leaf_check):
    value = _to_python(value)
    while isinstance(value, list) and value and not leaf_check(value):
        if isinstance(value[0], list):
            value = value[0] if len(value) == 1 else max(value, key=len)
        else:
            break
    return value if isinstance(value, list) else []


def flatten_atom_names(value) -> list[list[str]]:
    def is_residues(tree):
        return (
            isinstance(tree, list)
            and tree
            and isinstance(tree[0], list)
            and (not tree[0] or isinstance(tree[0][0], str))
        )

    atoms = _per_residue_lists(value, is_residues)
    out = []
    for residue in atoms:
        if not isinstance(residue, list):
            out.append(["CA"])
            continue
        names = [str(name) for name in residue if name is not None]
        out.append(names or ["CA"])
    return out


def flatten_atom_pos(value) -> list[list[list[float]]]:
    def is_residues(tree):
        return (
            isinstance(tree, list)
            and tree
            and isinstance(tree[0], list)
            and (not tree[0] or _is_xyz(tree[0][0]))
        )

    coords = _per_residue_lists(value, is_residues)
    out = []
    for residue in coords:
        if not isinstance(residue, list) or not residue:
            out.append([[0.0, 0.0, 0.0]])
            continue
        atoms = []
        for atom in residue:
            if _is_xyz(atom):
                atoms.append([float(atom[0]), float(atom[1]), float(atom[2])])
        out.append(atoms or [[0.0, 0.0, 0.0]])
    return out


def flatten_residue_numbers(value, n_res: int) -> list[int]:
    value = _longest_or_unwrap(_to_python(value))
    if isinstance(value, list) and value and not isinstance(value[0], list):
        numbers = []
        for item in value:
            try:
                numbers.append(int(item))
            except (TypeError, ValueError):
                numbers.append(len(numbers) + 1)
        if len(numbers) >= n_res:
            return numbers[:n_res]
    return list(range(1, n_res + 1))


def as_protease_record(
    record: ak.Record,
    molecule_id: str | None = None,
    residue_slice: tuple[int, int] | None = None,
    residue_labels: list[str] | None = None,
) -> ak.Record:
    """Rebuild a protease fold as scene / frame / chain / residue / atom."""
    letters = residue_letters_from(record["residue_label"])
    atoms = (
        flatten_atom_names(record["atom_label"])
        if "atom_label" in record.fields
        else [["CA"] for _ in letters]
    )
    pos = (
        flatten_atom_pos(record["atom_pos"])
        if "atom_pos" in record.fields
        else [[[float(i), 0.0, 0.0]] for i in range(len(letters))]
    )
    n = min(len(letters), len(atoms), len(pos))
    letters, atoms, pos = letters[:n], atoms[:n], pos[:n]
    numbers = flatten_residue_numbers(
        record["residue_number"] if "residue_number" in record.fields else None,
        n,
    )[:n]
    if residue_slice is not None:
        start, end = residue_slice
        letters, atoms, pos, numbers = (
            letters[start:end],
            atoms[start:end],
            pos[start:end],
            numbers[start:end],
        )
    if residue_labels is not None:
        n = min(len(residue_labels), len(atoms), len(pos))
        letters, atoms, pos, numbers = (
            residue_labels[:n],
            atoms[:n],
            pos[:n],
            list(range(1, n + 1)),
        )
    if molecule_id is None:
        molecule_id = "protease"
        if "molecule_id" in record.fields:
            ident = _to_python(record["molecule_id"])
            while isinstance(ident, list) and ident:
                ident = ident[0]
            if ident is not None:
                molecule_id = str(ident)
    data = {
        "chain_label": ak.Array(["A"]),
        "residue_label": ak.Array([letters]),
        "residue_number": ak.Array([numbers]),
        "atom_label": ak.Array([atoms]),
        "atom_pos": ak.Array([pos]),
    }
    if "residue_pLDDT" in record.fields:
        plddt = flatten_residue_numbers(record["residue_pLDDT"], len(letters))
        data["residue_pLDDT"] = ak.Array([[float(v) for v in plddt]])
    data = {key: value[np.newaxis, np.newaxis] for key, value in data.items()}
    data["molecule_id"] = ak.Array([[molecule_id]])
    return ak.Record(data)


def parse_structure(path: Path) -> ak.Record | None:
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    try:
        if ".cif" in suffixes:
            record = CifProcessor.process_file(path)
        else:
            record = PdbProcessor.process_file(path)
    except Exception as exc:
        note(f"Failed to parse {path.name}: {exc}")
        return None
    if record is None:
        return None
    try:
        return as_protease_record(record)
    except Exception as exc:
        note(f"Failed to normalize {path.name}: {exc}")
        return None


def set_molecule_id(record: ak.Record, molecule_id: str) -> ak.Record:
    data = {field: record[field] for field in record.fields}
    data["molecule_id"] = ak.Array([[molecule_id]])
    return ak.Record(data)


def best_pdb_id(uniprot: str, cache_dir: Path) -> str | None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{uniprot}.json"
    if path.exists():
        payload = load(path)
    else:
        try:
            response = requests.get(
                PDBE_BEST.format(uniprot=uniprot),
                timeout=30,
                headers={"User-Agent": "bioverse-merops"},
            )
            if not response.ok:
                save({}, path)
                return None
            payload = response.json()
            save(payload, path)
        except requests.RequestException:
            return None
    entries = None
    if isinstance(payload, dict):
        entries = (
            payload.get(uniprot)
            or payload.get(uniprot.upper())
            or payload.get(uniprot.lower())
        )
        if entries is None and len(payload) == 1:
            entries = next(iter(payload.values()))
    if not entries:
        return None
    pdb_id = str(entries[0].get("pdb_id", "")).lower()
    return pdb_id or None


def resolve_protease_structure(
    uniprot: str,
    *,
    pdb_dir: Path,
    af_dir: Path,
    mapping_dir: Path,
    molecule_id: str,
) -> tuple[ak.Record, str] | None:
    """Return a single-chain structure record and ``pdb`` / ``alphafold`` source."""
    pdb_id = best_pdb_id(uniprot, mapping_dir)
    if pdb_id:
        path = ensure_structure(pdb_id, pdb_dir, progress=False)
        if path is not None:
            record = parse_structure(path)
            if record is not None:
                return set_molecule_id(record, molecule_id), "pdb"
    path = fetch_alphafold_structure(uniprot, af_dir)
    if path is None:
        return None
    record = parse_structure(path)
    if record is None:
        return None
    return set_molecule_id(record, molecule_id), "alphafold"


def records_to_batches(records: list[ak.Record]):
    from ..data import Batch
    from ..utilities import IteratorWithLength, batched

    kept = []
    for record in records:
        try:
            Batch(ak.Array([record]))
        except Exception as exc:
            ident = "unknown"
            if "molecule_id" in record.fields:
                ident = ak.to_list(record["molecule_id"])
            note(f"Skipping unbatchable record {ident}: {exc}")
            continue
        kept.append(record)
    if not kept:
        raise RuntimeError("No batchable protease records.")
    return batched(IteratorWithLength(iter(kept), len(kept)))
