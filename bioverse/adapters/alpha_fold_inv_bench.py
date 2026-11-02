from __future__ import annotations

import awkward as ak
import numpy as np

from ..adapter import Adapter
from ..data import Assets, Split
from ..processors import PdbProcessor
from ..utilities import (
    IteratorWithLength,
    batched,
    config,
    download,
    glob_delete,
    load,
    note,
    parallelize,
    progressbar,
)
from ..utilities.id_mapping import (
    build_alphafold_index,
    ensure_protein_invbench,
    fetch_alphafold_structures,
    load_cath_domain_boundaries,
    load_sifts_segments,
    resolve_invbench_domain_mapping,
)


def crop_record_to_ranges(record: ak.Record, ranges: list[tuple[int, int]]):
    """Keep residues whose AlphaFold/UniProt numbers fall in ``ranges``."""
    if not ranges:
        return None
    data = {field: record[field] for field in record.fields}
    residue_numbers = np.asarray(ak.to_list(data["residue_number"][0, 0, 0]))
    mask = np.zeros(len(residue_numbers), dtype=bool)
    for start, end in ranges:
        mask |= (residue_numbers >= int(start)) & (residue_numbers <= int(end))
    if not mask.any():
        return None
    keep = np.flatnonzero(mask)
    for key, value in list(data.items()):
        prefix = key.split("_")[0]
        if prefix in {"residue", "atom"}:
            data[key] = value[:, :, :, keep]
    return ak.Record(data)


class AlphaFoldInvBenchAdapter(Adapter):
    """Download AlphaFold CATH domains matching the ProteinInvBench CATH 4.2 split."""

    @classmethod
    def download(
        cls,
        af_name: str = "swissprot_pdb",
        af_version: str = "v4",
        use_api_fallback: bool = True,
        api_workers: int | None = None,
    ):
        invbench_path = ensure_protein_invbench()

        af_path = config.raw_path / "AlphaFoldDB" / af_version / af_name
        if not af_path.exists() or not any(af_path.glob("AF-*")):
            base_url = "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/"
            download(f"{base_url}/{af_name}_{af_version}.tar", af_path)
            glob_delete(str(af_path / "*.cif.gz"))

        api_cache_dir = config.raw_path / "AlphaFoldDB" / "api"
        api_workers = api_workers or max(config.workers, 32)

        split_lookup = load(
            invbench_path / "data" / "cath4.2" / "chain_set_splits.json"
        )
        split_lookup = {
            name: ["train", "val", "test"][index]
            for index, key in enumerate(["train", "validation", "test"])
            for name in split_lookup[key]
        }
        proteins = load(invbench_path / "data" / "cath4.2" / "chain_set.jsonl")
        proteins = np.array([item for item in proteins if item["name"] in split_lookup])

        note("Loading SIFTS residue mapping and CATH 4.2 domain boundaries.")
        sifts_segments = load_sifts_segments()
        cath_boundaries = load_cath_domain_boundaries()
        note("Indexing AlphaFold Swiss-Prot structures.")
        alphafold_index = build_alphafold_index(af_path, version=af_version)

        pending_api = []
        tasks = []
        task_splits = []
        skipped = {
            "no_domain": [],
            "no_uniprot": [],
            "no_alphafold": [],
        }
        mapping_sources = {"local": 0, "api": 0}

        for item in progressbar(proteins, description="Mapping structures"):
            domain_mapping = resolve_invbench_domain_mapping(
                item["name"],
                sifts_segments,
                cath_boundaries,
                seq=item.get("seq"),
            )
            if not domain_mapping:
                skipped["no_domain"].append(item["name"])
                continue

            resolved = None
            for uniprot, ranges in domain_mapping:
                if uniprot in alphafold_index:
                    resolved = (uniprot, alphafold_index[uniprot], "local", ranges)
                    break

            if resolved is not None:
                tasks.append(
                    (item, resolved[0], resolved[1], resolved[2], resolved[3])
                )
                task_splits.append(split_lookup[item["name"]])
                mapping_sources["local"] += 1
            elif use_api_fallback:
                pending_api.append(
                    (item, domain_mapping, split_lookup[item["name"]])
                )
            else:
                skipped["no_alphafold"].append(item["name"])

        if use_api_fallback and pending_api:
            unique_uniprots = sorted(
                {
                    uniprot
                    for _, domain_mapping, _ in pending_api
                    for uniprot, _ in domain_mapping
                }
            )
            note(
                f"Fetching {len(unique_uniprots)} unique AlphaFold structures "
                f"for {len(pending_api)} domains ({api_workers} workers)."
            )
            api_index = fetch_alphafold_structures(
                unique_uniprots, api_cache_dir, workers=api_workers
            )

            for item, domain_mapping, partition in pending_api:
                resolved = None
                for uniprot, ranges in domain_mapping:
                    if uniprot in api_index:
                        resolved = (uniprot, api_index[uniprot], "api", ranges)
                        break
                if resolved is None:
                    skipped["no_alphafold"].append(item["name"])
                    continue
                tasks.append(
                    (item, resolved[0], resolved[1], resolved[2], resolved[3])
                )
                task_splits.append(partition)
                mapping_sources["api"] += 1

        note(
            f"Mapped {len(tasks)} / {len(proteins)} ProteinInvBench domains to AlphaFold "
            f"({mapping_sources['local']} local, {mapping_sources['api']} API)."
        )
        if not tasks:
            raise ValueError(
                "No ProteinInvBench domains could be mapped to AlphaFold structures. "
                f"Indexed {len(alphafold_index)} local AlphaFold files at {af_path}. "
                "Check that D_AFSP00 raw data is present and CATH/SIFTS mappings "
                "are available."
            )

        def process_protein(args):
            item, uniprot, structure_path, _source, ranges = args
            record = PdbProcessor.process_file(structure_path)
            cropped = crop_record_to_ranges(record, ranges)
            if cropped is None:
                return None
            data = {field: cropped[field] for field in cropped.fields}
            data["molecule_id"] = ak.Array([[item["name"]]])
            data["molecule_cath"] = ak.Array([[item["CATH"]]])
            data["molecule_uniprot"] = ak.Array([[uniprot]])
            data["molecule_structure_source"] = ak.Array([["alphafold"]])
            return ak.Record(data)

        raw_records = list(
            parallelize(
                process_protein,
                tasks,
                description="Processing AlphaFold domains",
            )
        )
        records = []
        kept_splits = []
        for record, task, partition in zip(raw_records, tasks, task_splits):
            if record is None:
                skipped["no_domain"].append(task[0]["name"])
                continue
            records.append(record)
            kept_splits.append(partition)
        if not records:
            raise ValueError(
                "CATH domain residue ranges did not match any AlphaFold structures."
            )
        partitions = ("train", "val", "test")
        rng = np.random.default_rng(config.seed)
        ordered_records = []
        ordered_splits = []
        for partition in partitions:
            bucket = [
                record
                for record, part in zip(records, kept_splits)
                if part == partition
            ]
            if bucket:
                order = rng.permutation(len(bucket))
                bucket = [bucket[int(i)] for i in order]
            ordered_records.extend(bucket)
            ordered_splits.extend([partition] * len(bucket))
        split = np.array(ordered_splits)
        batches = batched(
            IteratorWithLength(iter(ordered_records), len(ordered_records))
        )

        assets = Assets(
            {
                "mapping_summary": {
                    "total": int(len(proteins)),
                    "mapped": len(records),
                    "mapped_local": mapping_sources["local"],
                    "mapped_api": mapping_sources["api"],
                    "skipped_no_domain": len(skipped["no_domain"]),
                    "skipped_no_uniprot": len(skipped["no_uniprot"]),
                    "skipped_no_alphafold": len(skipped["no_alphafold"]),
                    "af_name": af_name,
                    "af_version": af_version,
                    "use_api_fallback": use_api_fallback,
                    "api_workers": api_workers,
                },
                "skipped": skipped,
                "split_sizes": {
                    partition: int(np.sum(split == partition))
                    for partition in partitions
                },
            }
        )

        return (
            batches,
            Split(
                {"ProteinInvBench_scene_split": split},
                default="ProteinInvBench_scene_split",
            ),
            assets,
        )
