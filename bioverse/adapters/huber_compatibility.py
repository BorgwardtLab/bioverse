"""Huber 2025 TEV DMS as a held-out protease–substrate compatibility benchmark."""

from __future__ import annotations

from pathlib import Path

import awkward as ak
import numpy as np

from ..adapter import Adapter
from ..data import Assets, Split
from ..transforms.structure_prediction import StructurePredictionTransform
from ..utilities import (
    IteratorWithLength,
    batched,
    config,
    download,
)
from ..utilities.cleavage import WINDOW_LEN, huber_window
from .huber_protease import (
    RAW_URL,
    TEVS_INDEX,
    WILDTYPE_PROTEASE,
    WILDTYPE_SUBSTRATE,
    _find_single_tsv,
    _valid_mask,
)
from .protease_structures import (
    as_protease_record,
    parse_structure,
    residue_letters_from,
    set_molecule_id,
)
from .biolip import ensure_structure

TEV_PDB_IDS = ("1lvm", "1lvb")


def _sequence_record(protease: str) -> ak.Record:
    data = {
        "chain_label": ak.Array(["A"]),
        "residue_label": ak.Array([list(protease)]),
        "residue_number": ak.Array([np.arange(1, len(protease) + 1)]),
    }
    data = {key: value[np.newaxis, np.newaxis] for key, value in data.items()}
    data["molecule_id"] = ak.Array([["TEVp"]])
    return ak.Record(data)


def _record_sequence(record: ak.Record) -> str:
    labels = ak.to_list(record["residue_label"])
    while isinstance(labels, list) and len(labels) == 1:
        labels = labels[0]
    if isinstance(labels, list) and labels and isinstance(labels[0], list):
        labels = max(labels, key=len)
    return "".join(str(ch) for ch in labels)


def _align_offset(query: str, subject: str, min_identity: float = 0.9):
    """Return (query_start, subject_start, length) for the best overlap."""
    if not query or not subject:
        return None
    best = (-1.0, 0, 0, 0)
    min_len = min(20, min(len(query), len(subject)))
    for q0 in range(len(query)):
        length = min(len(query) - q0, len(subject))
        if length < min_len:
            break
        ident = sum(a == b for a, b in zip(query[q0 : q0 + length], subject[:length]))
        score = ident / length
        if score > best[0] or (score == best[0] and length > best[3]):
            best = (score, q0, 0, length)
    for s0 in range(1, len(subject)):
        length = min(len(query), len(subject) - s0)
        if length < min_len:
            break
        ident = sum(a == b for a, b in zip(query[:length], subject[s0 : s0 + length]))
        score = ident / length
        if score > best[0] or (score == best[0] and length > best[3]):
            best = (score, 0, s0, length)
    score, q0, s0, length = best
    if length < min_len or score < min_identity:
        return None
    return q0, s0, length


def _crop_to_query(record: ak.Record, query: str) -> ak.Record | None:
    record = as_protease_record(record, molecule_id="TEVp")
    subject = "".join(residue_letters_from(record["residue_label"]))
    overlap = _align_offset(query, subject)
    if overlap is None:
        return None
    q0, s0, length = overlap
    return as_protease_record(
        record,
        molecule_id="TEVp",
        residue_slice=(s0, s0 + length),
        residue_labels=list(query[q0 : q0 + length]),
    )


def load_tev_structure(
    protease: str = WILDTYPE_PROTEASE,
    cache_dir: Path | None = None,
) -> tuple[ak.Record, str]:
    cache_dir = Path(cache_dir or (config.raw_path / "HUBCMP"))
    for pdb_id in TEV_PDB_IDS:
        path = ensure_structure(pdb_id, cache_dir / "pdb", progress=False)
        if path is None:
            continue
        record = parse_structure(path)
        if record is None:
            continue
        cropped = _crop_to_query(record, protease)
        if cropped is not None:
            return set_molecule_id(cropped, "TEVp"), f"pdb:{pdb_id}"
    return _sequence_record(protease), "sequence"


def compatibility_tables(
    df,
    protease: str = WILDTYPE_PROTEASE,
    substrate: str = WILDTYPE_SUBSTRATE,
):
    """Protease mutation sites, labels, AUC, and P6–P4′ substrate windows."""
    df = df[_valid_mask(df)].reset_index(drop=True)
    protease_mut = df["n_mut_TEVp"].to_numpy() == 1
    substrate_mut = df["n_mut_TEVs"].to_numpy() == 1
    tevp_pos = df["TEVp_pos"].to_numpy()
    tevp_aa = df["TEVp_AA"].astype(str).to_numpy()
    tevs_pos = df["TEVs_pos"].astype(str).to_numpy()
    tevs_aa = df["TEVs_AA"].astype(str).to_numpy()
    auc = df["AUC"].to_numpy(dtype=float)
    positions, labels, effects, windows, kinds = [], [], [], [], []
    for i in range(len(df)):
        pos, lab = [], []
        tevs = list(substrate)
        kind = "wt"
        if protease_mut[i]:
            pos.append(int(tevp_pos[i]) - 1)
            lab.append(tevp_aa[i])
            kind = "protease"
        if substrate_mut[i]:
            tevs[TEVS_INDEX[tevs_pos[i]]] = tevs_aa[i]
            kind = "both" if protease_mut[i] else "substrate"
        if not pos and not substrate_mut[i]:
            kind = "wt"
        positions.append(pos)
        labels.append(lab)
        effects.append(float(auc[i]))
        windows.append(huber_window("".join(tevs)))
        kinds.append(kind)
    return positions, labels, effects, windows, kinds


class HuberCompatibilityAdapter(Adapter):
    """Huber et al. TEV DMS as protease structure + substrate-window pairs.

    Huber 2025 is evaluation-only: every row is assigned to the test
    partition. The protease is the catalytic-domain sequence used in the
    screen, with a PDB TEV fold when it aligns and Boltz-2 as fallback.
    """

    @classmethod
    def download(
        cls,
        model: str = "boltz2",
        use_msa_server: bool = True,
        cache_dir: str | Path | None = None,
        predict_if_needed: bool = True,
    ):
        path = config.raw_path / "MLDEEP"
        download(RAW_URL, path)
        df = _find_single_tsv(path)
        positions, labels, effects, windows, kinds = compatibility_tables(df)
        record, source = load_tev_structure(
            WILDTYPE_PROTEASE, cache_dir=config.raw_path / "HUBCMP"
        )
        has_coords = "atom_pos" in record.fields
        record = ak.Record(
            {
                **{field: record[field] for field in record.fields},
                "molecule_mutation_label": [[labels]],
                "molecule_mutation_pos": [[positions]],
                "molecule_mutation_effect": [[effects]],
                "molecule_mutation_substrate": [[windows]],
                "molecule_mutation_kind": [[kinds]],
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
                "window_len": WINDOW_LEN,
                "structure_source": source,
            }
        )
        if has_coords or not predict_if_needed:
            return batches, split, assets
        transform = StructurePredictionTransform(
            model=model,
            use_msa_server=use_msa_server,
            cache_dir=cache_dir,
        )
        return transform(batches, split, assets)
