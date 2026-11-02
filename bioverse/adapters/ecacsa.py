"""Adapter merging ProteinShake EC data with M-CSA catalytic-site annotations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import awkward as ak
import numpy as np
import requests
from biopandas.mmcif import PandasMmcif
from fastavro import reader as avro_reader

from ..adapter import Adapter
from ..data import Assets, Split
from ..utilities import (
    THREE_TO_ONE,
    IteratorWithLength,
    batched,
    config,
    download,
    load,
    note,
    save,
)
from .proteinshake import REPOSITORY_URL, convert_protein

MCSA_ENTRIES_URL = (
    "https://www.ebi.ac.uk/thornton-srv/m-csa/api/entries/?format=json"
)
MCSA_RESIDUES_URL = (
    "https://www.ebi.ac.uk/thornton-srv/m-csa/api/residues/?format=json"
)
PDBE_ASSEMBLY_URL = "https://www.ebi.ac.uk/pdbe/api/pdb/entry/assembly/{pdb_id}"
UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
RCSB_ASSEMBLY_CIF = "https://files.rcsb.org/download/{pdb_id}-assembly1.cif"
RCSB_CIF = "https://files.rcsb.org/download/{pdb_id}.cif"
PDE_CIF = "https://www.ebi.ac.uk/pdbe/entry-files/download/{pdb_id}.cif"

THREE_LETTER = {v: k for k, v in THREE_TO_ONE.items()}


def _parse_fasta_sequence(text: str) -> str:
    return "".join(line.strip() for line in text.splitlines() if not line.startswith(">"))


def _ec_fields(ec: str) -> dict[str, ak.Array]:
    levels = [part for part in ec.split(".") if part]
    data = {
        "molecule_ec": ak.Array([[ec]]),
    }
    for i in range(1, len(levels) + 1):
        data[f"molecule_ec{i}"] = ak.Array([[ ".".join(levels[:i]) ]])
    return data


def _fetch_paginated(url: str, cache_path: Path) -> list[dict]:
    if cache_path.exists():
        return load(cache_path)

    results: list[dict] = []
    page = 1
    while True:
        response = requests.get(
            url,
            params={"page": page, "page_size": 500},
            timeout=120,
            headers={"User-Agent": "bioverse-ecacsa"},
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            save(payload, cache_path)
            return payload
        results.extend(payload["results"])
        if payload.get("next") is None:
            break
        page += 1

    save(results, cache_path)
    return results


def _fetch_assembly_chains(pdb_id: str, cache: dict[str, list[str]]) -> list[str]:
    pdb_id = pdb_id.lower()
    if pdb_id in cache:
        return cache[pdb_id]

    chains: list[str] = []
    try:
        response = requests.get(
            PDBE_ASSEMBLY_URL.format(pdb_id=pdb_id),
            timeout=60,
            headers={"User-Agent": "bioverse-ecacsa"},
        )
        if response.ok:
            payload = response.json().get(pdb_id, [])
            if payload:
                for entity in payload[0].get("entities", []):
                    if str(entity.get("molecule_type", "")).startswith("polypeptide"):
                        chains.extend(entity.get("in_chains", []))
    except requests.RequestException:
        pass

    cache[pdb_id] = chains
    return chains


def _resolve_assembly_chain(
    assembly_chain_name: str,
    chain_name: str,
    protein_chains: list[str],
    asym_ids: Iterable[str],
    auth_by_asym: dict[str, str],
) -> str | None:
    asym_ids = list(asym_ids)
    if assembly_chain_name in asym_ids:
        return assembly_chain_name
    if chain_name in asym_ids:
        return chain_name

    if protein_chains:
        if assembly_chain_name in protein_chains:
            return assembly_chain_name
        base = assembly_chain_name[0] if assembly_chain_name else chain_name
        copy_idx = max(len(assembly_chain_name) - 1, 0)
        matches = []
        for asym in protein_chains:
            if asym not in asym_ids:
                continue
            auth = auth_by_asym.get(asym, asym)
            if auth == base or str(asym).startswith(base):
                matches.append(asym)
        if copy_idx < len(matches):
            return matches[copy_idx]

    auth_matches = [
        asym
        for asym in asym_ids
        if auth_by_asym.get(asym, asym) == chain_name
    ]
    if len(auth_matches) == 1:
        return auth_matches[0]
    return None


def _residue_code_matches(label_comp_id: str, code: str) -> bool:
    one = THREE_TO_ONE.get(label_comp_id.upper(), label_comp_id[:1])
    expected = THREE_TO_ONE.get(code.upper(), code[:1])
    return one == expected


def _match_catalytic_residue(
    df,
    asym: str,
    resid: int,
    auth_resid: int,
    code: str,
) -> tuple[int, str] | None:
    sub = df[df["label_asym_id"] == asym]
    if sub.empty:
        return None
    attempts = (
        ("label_seq_id", resid),
        ("auth_seq_id", auth_resid),
        ("label_seq_id", auth_resid),
        ("auth_seq_id", resid),
    )
    seen = set()
    for column, number in attempts:
        key = (column, int(number))
        if key in seen:
            continue
        seen.add(key)
        hits = sub[sub[column].astype(int) == int(number)]
        if hits.empty:
            continue
        row = hits.iloc[0]
        if _residue_code_matches(row["label_comp_id"], code):
            return int(row["label_seq_id"]), row["label_comp_id"]
    return None


def _download_structure(pdb_id: str, structures_dir: Path) -> Path | None:
    pdb_id = pdb_id.lower()
    for url_template in (RCSB_ASSEMBLY_CIF, RCSB_CIF, PDE_CIF):
        out_path = structures_dir / f"{pdb_id}.cif"
        if out_path.exists():
            return out_path
        url = url_template.format(pdb_id=pdb_id)
        try:
            download(url, out_path)
            return out_path
        except Exception:
            if out_path.exists():
                out_path.unlink(missing_ok=True)
    return None


def _structure_dataframe(path: Path):
    mmcif = PandasMmcif().read_mmcif(str(path))
    df = mmcif.df["ATOM"].copy()
    df = df[df["group_PDB"] == "ATOM"]
    return df


def _cif_to_record(
    df,
    chain_asym_ids: list[str] | None = None,
) -> dict[str, ak.Array]:
    if chain_asym_ids:
        df = df[df["label_asym_id"].isin(chain_asym_ids)]
    if df.empty:
        raise ValueError("No atoms left after chain filtering.")

    residue_sizes = ak.Array(df["label_seq_id"].astype(int).to_numpy()).run_lengths()
    chain_sizes = ak.Array(df["label_asym_id"].to_list()).run_lengths()
    data = {
        "chain_label": df["label_asym_id"].to_list(),
        "residue_number": df["label_seq_id"].astype(int).to_numpy(),
        "residue_label": [
            THREE_TO_ONE.get(residue, residue)
            for residue in df["label_comp_id"].to_list()
        ],
        "atom_label": df["label_atom_id"].to_list(),
        "atom_pos": np.stack(
            (
                df["Cartn_x"].astype(np.float32).to_numpy(),
                df["Cartn_y"].astype(np.float32).to_numpy(),
                df["Cartn_z"].astype(np.float32).to_numpy(),
            ),
            axis=1,
        ),
    }
    data = {
        k: ak.Array(v).unflatten(chain_sizes).unflatten(residue_sizes, 1)
        for k, v in data.items()
    }
    data["residue_number"] = data["residue_number"].firsts(2)
    data["residue_label"] = data["residue_label"].firsts(2)
    data["chain_label"] = data["chain_label"].firsts(2).firsts(1)
    return data


def _annotate_catalytic_sites(
    data: dict[str, ak.Array],
    catalytic_keys: set[tuple[str, int]],
) -> ak.Array:
    chain_labels = ak.to_list(data["chain_label"])
    residue_numbers = ak.to_list(data["residue_number"])
    labels = []
    for chain, residues in zip(chain_labels, residue_numbers):
        chain_labels_inner = [chain] * len(residues)
        labels.append(
            [
                int((chain_id, int(res_num)) in catalytic_keys)
                for chain_id, res_num in zip(chain_labels_inner, residues)
            ]
        )
    return ak.Array(labels)


def _collect_entry_residues(entry: dict) -> list[dict]:
    residues = []
    for residue in entry.get("residues", []):
        for chain in residue.get("residue_chains", []):
            if chain.get("is_reference"):
                residues.append(chain)
    return residues


def _map_entry_catalytic_sites(
    entry: dict,
    df,
    assembly_cache: dict[str, list[str]],
) -> tuple[set[tuple[str, int]], dict]:
    pdb_id = None
    for chain in _collect_entry_residues(entry):
        pdb_id = chain["pdb_id"].lower()
        break
    if pdb_id is None:
        return set(), {"status": "no_reference_residues"}

    protein_chains = _fetch_assembly_chains(pdb_id, assembly_cache)
    asym_ids = list(df["label_asym_id"].unique())
    auth_by_asym = {
        asym: str(df[df["label_asym_id"] == asym]["auth_asym_id"].iloc[0])
        for asym in asym_ids
    }

    catalytic_keys: set[tuple[str, int]] = set()
    mapped = 0
    total = 0
    for chain in _collect_entry_residues(entry):
        total += 1
        asym = _resolve_assembly_chain(
            chain["assembly_chain_name"],
            chain.get("chain_name", chain["assembly_chain_name"]),
            protein_chains,
            asym_ids,
            auth_by_asym,
        )
        if asym is None:
            continue
        auth_resid = chain.get("auth_resid")
        auth_resid = int(auth_resid) if auth_resid is not None else int(chain["resid"])
        match = _match_catalytic_residue(
            df,
            asym,
            int(chain["resid"]),
            auth_resid,
            chain["code"],
        )
        if match is not None:
            catalytic_keys.add((asym, match[0]))
            mapped += 1

    return catalytic_keys, {
        "status": "ok" if mapped == total and total > 0 else "partial",
        "mapped": mapped,
        "total": total,
        "pdb_id": pdb_id,
    }


def _build_mcsa_record(
    entry: dict,
    cif_path: Path,
    assembly_cache: dict[str, list[str]],
) -> tuple[ak.Record | None, dict]:
    df = _structure_dataframe(cif_path)
    catalytic_keys, mapping_info = _map_entry_catalytic_sites(entry, df, assembly_cache)
    if mapping_info["status"] != "ok":
        return None, mapping_info

    pdb_id = mapping_info["pdb_id"]
    protein_chains = _fetch_assembly_chains(pdb_id, assembly_cache)
    asym_ids = list(df["label_asym_id"].unique())
    if protein_chains:
        chain_asym_ids = [asym for asym in protein_chains if asym in asym_ids]
    else:
        chain_asym_ids = [
            asym
            for asym in asym_ids
            if df[df["label_asym_id"] == asym]["group_PDB"].iloc[0] == "ATOM"
        ]
    data = _cif_to_record(df, chain_asym_ids=chain_asym_ids or None)
    data["residue_catalytic_site"] = _annotate_catalytic_sites(data, catalytic_keys)

    mcsa_id = entry["mcsa_id"]
    data["molecule_id"] = ak.Array([[f"{pdb_id.upper()}_M{mcsa_id}"]])
    data["molecule_source"] = ak.Array([["mcsa"]])
    data["molecule_mcsa_id"] = ak.Array([[mcsa_id]])
    uniprot_id = entry.get("reference_uniprot_id", "")
    data["molecule_uniprot_id"] = ak.Array([[uniprot_id]])

    ecs = entry.get("all_ecs") or []
    ec = ecs[0] if ecs else ""
    data |= _ec_fields(ec)
    data["molecule_enzyme_name"] = ak.Array([[entry.get("enzyme_name", "")]])

    record = ak.Record({k: v[np.newaxis, np.newaxis] for k, v in data.items()})
    mapping_info["catalytic_count"] = int(ak.sum(record.residue_catalytic_site))
    return record, mapping_info


def _load_uniprot_sequences(
    uniprot_ids: Iterable[str],
    cache_path: Path,
) -> dict[str, str]:
    if cache_path.exists():
        return load(cache_path)

    sequences = {}
    for uniprot_id in sorted(set(uniprot_ids)):
        if not uniprot_id:
            continue
        try:
            response = requests.get(
                UNIPROT_FASTA_URL.format(uniprot_id=uniprot_id),
                timeout=60,
                headers={"User-Agent": "bioverse-ecacsa"},
            )
            if response.ok:
                sequences[uniprot_id] = _parse_fasta_sequence(response.text)
        except requests.RequestException:
            continue

    save(sequences, cache_path)
    return sequences


def _load_proteinshake_records(
    avro_path: Path,
    excluded_sequences: set[str],
    excluded_ids: set[str],
) -> tuple[list[ak.Record], dict]:
    records = []
    stats = {"loaded": 0, "excluded_id": 0, "excluded_sequence": 0}
    with open(avro_path, "rb") as file:
        for protein in avro_reader(file):
            meta = protein["protein"]
            protein_id = meta["ID"]
            if protein_id in excluded_ids:
                stats["excluded_id"] += 1
                continue
            if meta["sequence"] in excluded_sequences:
                stats["excluded_sequence"] += 1
                continue
            record = convert_protein(protein)
            record_dict = {k: record[k] for k in record.fields}
            record_dict["molecule_source"] = ak.Array([["proteinshake"]])
            record_dict["molecule_mcsa_id"] = ak.Array([[-1]])
            record_dict["molecule_uniprot_id"] = ak.Array([[""]])
            record_dict["residue_catalytic_site"] = ak.zeros_like(
                record.residue_number, dtype=np.int64
            )
            records.append(ak.Record(record_dict))
            stats["loaded"] += 1
    return records, stats


class EcacsaAdapter(Adapter):
    """Merge ProteinShake EC structures with M-CSA catalytic-site test structures."""

    @classmethod
    def download(cls, val_size: int = 1000, seed: int | None = None):
        seed = config.seed if seed is None else seed
        rng = np.random.default_rng(seed)

        raw_root = config.raw_path / "M-CSA"
        raw_root.mkdir(parents=True, exist_ok=True)
        structures_dir = raw_root / "structures"
        structures_dir.mkdir(parents=True, exist_ok=True)

        note("Fetching M-CSA entries and curated residues from API.")
        entries = _fetch_paginated(MCSA_ENTRIES_URL, raw_root / "entries.json")
        _fetch_paginated(MCSA_RESIDUES_URL, raw_root / "residues.json")

        uniprot_ids = {
            entry.get("reference_uniprot_id", "")
            for entry in entries
            if entry.get("reference_uniprot_id")
        }
        uniprot_sequences = _load_uniprot_sequences(
            uniprot_ids,
            raw_root / "uniprot_sequences.json",
        )
        test_sequences = set(uniprot_sequences.values())

        assembly_cache: dict[str, list[str]] = {}
        mcsa_records = []
        mcsa_stats = {
            "accepted": 0,
            "rejected_mapping": 0,
            "rejected_download": 0,
            "rejected_empty": 0,
        }
        mcsa_pdb_ids: set[str] = set()

        note(f"Building M-CSA test structures ({len(entries)} entries).")
        for entry in entries:
            reference_chains = _collect_entry_residues(entry)
            if not reference_chains:
                mcsa_stats["rejected_empty"] += 1
                continue
            pdb_id = reference_chains[0]["pdb_id"].lower()
            mcsa_pdb_ids.add(pdb_id.upper())

            cif_path = _download_structure(pdb_id, structures_dir)
            if cif_path is None:
                mcsa_stats["rejected_download"] += 1
                continue

            try:
                record, mapping_info = _build_mcsa_record(
                    entry, cif_path, assembly_cache
                )
            except Exception:
                mcsa_stats["rejected_mapping"] += 1
                continue

            if record is None:
                mcsa_stats["rejected_mapping"] += 1
                continue

            mcsa_records.append(record)
            mcsa_stats["accepted"] += 1

        note(
            "M-CSA structures accepted: "
            f"{mcsa_stats['accepted']} / {len(entries)} "
            f"(download failures: {mcsa_stats['rejected_download']}, "
            f"mapping failures: {mcsa_stats['rejected_mapping']})."
        )

        ps_path = config.raw_path / "ProteinShake" / "EnzymeCommissionDataset"
        avro = ps_path / "EnzymeCommissionDataset.atom.avro"
        if not avro.exists():
            download(
                f"{REPOSITORY_URL}/EnzymeCommissionDataset.atom.avro.gz",
                ps_path / "EnzymeCommissionDataset.atom",
            )

        note("Loading ProteinShake EnzymeCommissionDataset with deduplication.")
        ps_records, ps_stats = _load_proteinshake_records(
            avro,
            excluded_sequences=test_sequences,
            excluded_ids=mcsa_pdb_ids,
        )
        note(
            "ProteinShake loaded: "
            f"{ps_stats['loaded']} kept, "
            f"{ps_stats['excluded_sequence']} removed by sequence overlap, "
            f"{ps_stats['excluded_id']} removed by PDB overlap."
        )

        indices = np.arange(len(ps_records))
        rng.shuffle(indices)
        val_count = min(val_size, len(ps_records))
        val_indices = set(indices[:val_count].tolist())
        split_labels = []
        ordered_records = []

        for idx, record in enumerate(ps_records):
            split_labels.append("val" if idx in val_indices else "train")
            ordered_records.append(record)

        split_labels.extend(["test"] * len(mcsa_records))
        ordered_records.extend(mcsa_records)

        split = Split(
            {"ecacsa_scene_split": split_labels},
            default="ecacsa_scene_split",
        )
        assets = Assets(
            {
                "mcsa_stats": mcsa_stats,
                "proteinshake_stats": ps_stats,
                "test_uniprot_ids": sorted(uniprot_ids),
                "split_seed": seed,
                "val_size": val_count,
            }
        )

        return (
            batched(IteratorWithLength(iter(ordered_records), len(ordered_records))),
            split,
            assets,
        )
