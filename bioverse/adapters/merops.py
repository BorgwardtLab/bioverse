"""MEROPS experimentally observed protease cleavage sites as weak supervision."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import awkward as ak
import numpy as np

from ..adapter import Adapter
from ..data import Assets, Split
from ..utilities import (
    PROTEIN_ALPHABET,
    config,
    download,
    note,
)
from ..utilities.cleavage import (
    WINDOW_LEN,
    allowed_letters,
    canonical_count,
    merops_p4p4_window,
    pad_window,
    prefer_uniprot,
    residue_letter,
)
from .protease_structures import records_to_batches, resolve_protease_structure

MEROPS_FTP = "https://ftp.ebi.ac.uk/pub/databases/merops/current_release/database_files"
SUBSTRATE_FILE = "Substrate_search.txt"
UNIPROT_FILE = "MID_2_UniProt.txt"
COMBINATORIAL_FILE = "combinatorial_substrates.txt"
DEFAULT_EXCLUDE = ("C04.004", "M10.003", "M10.004")
_CODE_RE = re.compile(r"^[A-Z]\d{2}\.\d{3}$")


def merops_path() -> Path:
    return config.raw_path / "MEROPS"


def _strip(cell) -> str:
    return str(cell).strip().strip("'").strip('"')


def _ensure_file(name: str) -> Path:
    dest = merops_path() / name
    download(f"{MEROPS_FTP}/{name}", dest)
    if dest.exists():
        return dest
    matches = list(dest.parent.glob(f"{dest.stem}*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"MEROPS file {name} was not downloaded to {dest}.")


def load_table(name: str) -> list[list[str]]:
    path = _ensure_file(name)
    text = path.read_text(encoding="utf-8", errors="replace")
    sample = text[:4096]
    delimiter = "," if sample.count(",") > sample.count("\t") else "\t"
    rows = []
    for row in csv.reader(text.splitlines(), delimiter=delimiter):
        if row:
            rows.append([_strip(cell) for cell in row])
    return rows


def load_uniprot_map(rows: list[list[str]] | None = None) -> dict[str, list[str]]:
    rows = load_table(UNIPROT_FILE) if rows is None else rows
    mapping: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if len(row) < 3 or row[0].lower() == "code":
            continue
        mapping[row[0]].append(row[2] if len(row) > 2 else "")
    return mapping


def parse_cleavages(
    rows: list[list[str]] | None = None,
    *,
    min_canonical: int = 4,
    require_p1: bool = True,
) -> dict[str, list[str]]:
    """Return unique P6–P4′ windows per MEROPS peptidase identifier."""
    rows = load_table(SUBSTRATE_FILE) if rows is None else rows
    windows: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if len(row) < 12:
            continue
        code = row[1]
        if not _CODE_RE.match(code):
            continue
        window = merops_p4p4_window(row[4:12])
        if require_p1 and residue_letter(row[7]) not in PROTEIN_ALPHABET:
            continue
        if canonical_count(window) < min_canonical:
            continue
        if window in seen[code]:
            continue
        seen[code].add(window)
        windows[code].append(window)
    return windows


def parse_combinatorial_profiles(
    rows: list[list[str]] | None = None,
) -> dict[str, list[dict]]:
    rows = load_table(COMBINATORIAL_FILE) if rows is None else rows
    profiles: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if len(row) < 10 or not _CODE_RE.match(row[0]):
            continue
        allowed = [allowed_letters(cell) for cell in row[2:10]]
        restricted = sum(site is not None for site in allowed)
        if restricted == 0:
            continue
        consensus = []
        for site in allowed:
            consensus.append(site[0] if site else "-")
        profiles[row[0]].append(
            {
                "allowed": allowed,
                "consensus": pad_window("".join(consensus), align="p4"),
                "restricted": restricted,
                "organism": row[13] if len(row) > 13 else "",
            }
        )
    return profiles


def expand_profile_windows(profile: dict, alphabet: str = PROTEIN_ALPHABET):
    """Yield (window, label) pairs for single-site variants of a consensus."""
    consensus = list(profile["consensus"])
    # Combinatorial tables are P4–P4′; the shared window left-pads P6–P5.
    offset = 2
    for local_i, allowed in enumerate(profile["allowed"]):
        if allowed is None:
            continue
        site = offset + local_i
        if site >= len(consensus):
            continue
        allowed_set = set(allowed)
        wt = consensus[site]
        for aa in alphabet:
            variant = consensus.copy()
            variant[site] = aa
            yield "".join(variant), 1.0 if aa in allowed_set else 0.0, wt, aa, site


def _attach_pairs(record, windows: list[str], effects: list[float] | None = None):
    effects = [1.0] * len(windows) if effects is None else effects
    n = len(windows)
    data = {field: record[field] for field in record.fields}
    data["molecule_mutation_label"] = [[[""] * n]]
    data["molecule_mutation_pos"] = [[[-1] * n]]
    data["molecule_mutation_effect"] = [[list(map(float, effects))]]
    data["molecule_mutation_substrate"] = [[list(map(str, windows))]]
    return ak.Record(data)


def _family(code: str) -> str:
    return code.split(".", 1)[0]


def _window_split_counts(n: int, train: float = 0.8, val: float = 0.1) -> tuple[int, int, int]:
    """Train/val/test counts for one protease's windows."""
    n = int(n)
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    n_train = min(max(n_train, 1), n - 1)
    remaining = n - n_train
    n_val = min(max(n_val, 1), remaining)
    if n_val == remaining and remaining > 1:
        n_val = remaining - 1
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def _assign_window_splits(
    n_windows: list[int],
    codes: list[str],
    *,
    exclude: set[str],
    seed: int,
    train: float,
    val: float,
) -> list[str]:
    rng = np.random.default_rng(seed)
    labels: list[str] = []
    for count, code in zip(n_windows, codes):
        count = int(count)
        if code in exclude:
            labels.extend(["heldout"] * count)
            continue
        n_train, n_val, n_test = _window_split_counts(count, train, val)
        assignment = np.array(
            ["train"] * n_train + ["val"] * n_val + ["test"] * n_test,
            dtype=object,
        )
        labels.extend(assignment[rng.permutation(count)].tolist())
    return labels


def _assign_splits(
    codes: list[str],
    *,
    exclude: set[str],
    seed: int,
    train: float,
    val: float,
) -> list[str]:
    rng = np.random.default_rng(seed)
    labels = []
    remaining = []
    remaining_idx = []
    for i, code in enumerate(codes):
        if code in exclude:
            labels.append("heldout")
        else:
            labels.append(None)
            remaining.append(code)
            remaining_idx.append(i)
    order = rng.permutation(len(remaining))
    n = len(remaining)
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    assignment = np.array(["test"] * n, dtype=object)
    assignment[:n_train] = "train"
    assignment[n_train : n_train + n_val] = "val"
    assignment = assignment[np.argsort(order)]
    for local, idx in enumerate(remaining_idx):
        labels[idx] = str(assignment[local])
    return labels


class MeropsAdapter(Adapter):
    """Observed MEROPS cleavage windows paired with PDB/AlphaFold protease folds.

    Each scene is one peptidase structure. Observed P4–P4′ windows (padded to
    P6–P4′) are stored as ``molecule_mutation_*`` rows so
    :class:`~bioverse.samplers.mutation.MutationSampler` reuses the structure.
    Huber TEV (C04.004) and Cieplak MMPs (M10.003, M10.004) are held out.
    The default ``random_mutation_split`` assigns windows of each protease
    independently so train and val share enzymes.
    """

    @classmethod
    def download(
        cls,
        source: str = "cleavage",
        exclude_codes: list[str] | tuple[str, ...] | None = None,
        max_length: int = 400,
        min_canonical: int = 4,
        max_windows: int | None = 64,
        max_proteases: int | None = None,
        seed: int = 0,
        train_size: float = 0.8,
        val_size: float = 0.1,
        min_windows: int = 1,
    ):
        exclude = set(DEFAULT_EXCLUDE if exclude_codes is None else exclude_codes)
        uniprot_map = load_uniprot_map()
        if source == "combinatorial":
            max_windows = None if max_windows == 64 else max_windows
            profiles = parse_combinatorial_profiles()
            pairs = {}
            effects = {}
            for code, items in profiles.items():
                item = max(items, key=lambda row: row["restricted"])
                expanded = list(expand_profile_windows(item))
                if not expanded:
                    continue
                windows, labels, _, _, _ = zip(*expanded)
                pairs[code] = list(windows)
                effects[code] = list(labels)
        else:
            pairs = parse_cleavages(min_canonical=min_canonical)
            effects = {code: [1.0] * len(windows) for code, windows in pairs.items()}

        codes = sorted(code for code in pairs if code in uniprot_map)
        records = []
        used_codes = []
        n_windows = []
        pdb_dir = merops_path() / "pdb"
        af_dir = merops_path() / "alphafold"
        mapping_dir = merops_path() / "pdbe"
        for code in codes:
            if max_proteases is not None and len(records) >= max_proteases:
                break
            uniprot = prefer_uniprot(uniprot_map[code])
            if uniprot is None:
                continue
            try:
                resolved = resolve_protease_structure(
                    uniprot,
                    pdb_dir=pdb_dir,
                    af_dir=af_dir,
                    mapping_dir=mapping_dir,
                    molecule_id=code,
                )
            except Exception as exc:
                note(f"Failed structure for {code} ({uniprot}): {exc}")
                continue
            if resolved is None:
                note(f"No structure for {code} ({uniprot}).")
                continue
            record, source_name = resolved
            try:
                n_res = len(_peel_sequence(record))
            except Exception as exc:
                note(f"Unreadable sequence for {code} ({uniprot}): {exc}")
                continue
            if n_res == 0 or n_res > max_length:
                continue
            windows = pairs[code]
            if max_windows is not None:
                windows = windows[: max_windows]
            if len(windows) < min_windows:
                continue
            scores = effects[code][: len(windows)]
            try:
                record = _attach_pairs(record, windows, scores)
            except Exception as exc:
                note(f"Failed to attach windows for {code}: {exc}")
                continue
            data = {field: record[field] for field in record.fields}
            data["molecule_uniprot"] = ak.Array([[uniprot]])
            data["molecule_structure_source"] = ak.Array([[source_name]])
            data["molecule_merops_id"] = ak.Array([[code]])
            data["molecule_family"] = ak.Array([[_family(code)]])
            record = ak.Record(data)
            try:
                from ..data import Batch

                Batch(ak.Array([record]))
            except Exception as exc:
                note(f"Skipping {code}: cannot batch ({exc})")
                continue
            records.append(record)
            used_codes.append(code)
            n_windows.append(len(windows))

        if not records:
            raise RuntimeError("MEROPS adapter produced no protease structures.")

        if source == "combinatorial":
            scene_splits = ["test"] * len(used_codes)
            window_splits = ["test"] * int(sum(n_windows))
        else:
            scene_splits = _assign_splits(
                used_codes,
                exclude=exclude,
                seed=seed,
                train=train_size,
                val=val_size,
            )
            window_splits = _assign_window_splits(
                n_windows,
                used_codes,
                exclude=exclude,
                seed=seed,
                train=train_size,
                val=val_size,
            )
        batches = records_to_batches(records)
        split = Split(
            {
                "random_scene_split": scene_splits,
                "random_mutation_split": window_splits,
            },
            default="random_mutation_split",
        )
        assets = Assets(
            {
                "window_len": WINDOW_LEN,
                "exclude_codes": sorted(exclude),
                "source": source,
                "n_proteases": len(records),
            }
        )
        return batches, split, assets


def _sequence_letter(token) -> str:
    letter = residue_letter(token)
    return letter if letter in PROTEIN_ALPHABET else "X"


def _join_residues(labels) -> str:
    if isinstance(labels, str):
        return "".join(_sequence_letter(ch) for ch in labels)
    if not isinstance(labels, list) or not labels:
        return _sequence_letter(labels)
    if isinstance(labels[0], list):
        return _join_residues(max(labels, key=len))
    return "".join(_sequence_letter(token) for token in labels)


def _peel_sequence(record) -> str:
    labels = ak.to_list(record["residue_label"])
    while isinstance(labels, list) and len(labels) == 1:
        labels = labels[0]
    return _join_residues(labels)
