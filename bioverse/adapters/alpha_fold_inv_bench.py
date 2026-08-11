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
    fetch_alphafold_structures,
    load_sifts_mapping,
    parse_invbench_name,
)


class AlphaFoldInvBenchAdapter(Adapter):
    """Download AlphaFold structures curated for the inverse folding benchmark suite."""

    @classmethod
    def download(
        cls,
        af_name: str = "swissprot_pdb",
        af_version: str = "v4",
        use_api_fallback: bool = True,
        api_workers: int | None = None,
    ):
        invbench_path = config.raw_path / "ProteinInvBench"
        download(
            "https://github.com/A4Bio/ProteinInvBench/releases/download/dataset_release/data.tar.gz",
            invbench_path,
        )

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
        proteins = np.array(
            [item for item in proteins if item["name"] in split_lookup]
        )

        note("Loading SIFTS PDB-to-UniProt mapping.")
        sifts_mapping = load_sifts_mapping()
        note("Indexing AlphaFold Swiss-Prot structures.")
        alphafold_index = build_alphafold_index(af_path, version=af_version)

        pending_api = []
        tasks = []
        task_splits = []
        skipped = {
            "no_uniprot": [],
            "no_alphafold": [],
        }
        mapping_sources = {"local": 0, "api": 0}

        for item in progressbar(proteins, description="Mapping structures"):
            pdb_id, chain = parse_invbench_name(item["name"])
            uniprots = sifts_mapping.get((pdb_id, chain), [])
            if not uniprots:
                skipped["no_uniprot"].append(item["name"])
                continue

            resolved = None
            for uniprot in uniprots:
                if uniprot in alphafold_index:
                    resolved = (uniprot, alphafold_index[uniprot], "local")
                    break

            if resolved is not None:
                tasks.append((item, resolved[0], resolved[1], resolved[2]))
                task_splits.append(split_lookup[item["name"]])
                mapping_sources["local"] += 1
            elif use_api_fallback:
                pending_api.append(
                    (item, uniprots, split_lookup[item["name"]])
                )
            else:
                skipped["no_alphafold"].append(item["name"])

        if use_api_fallback and pending_api:
            unique_uniprots = sorted(
                {uniprot for _, uniprots, _ in pending_api for uniprot in uniprots}
            )
            note(
                f"Fetching {len(unique_uniprots)} unique AlphaFold structures "
                f"for {len(pending_api)} chains ({api_workers} workers)."
            )
            api_index = fetch_alphafold_structures(
                unique_uniprots, api_cache_dir, workers=api_workers
            )

            for item, uniprots, partition in pending_api:
                resolved = None
                for uniprot in uniprots:
                    if uniprot in api_index:
                        resolved = (uniprot, api_index[uniprot], "api")
                        break
                if resolved is None:
                    skipped["no_alphafold"].append(item["name"])
                    continue
                tasks.append((item, resolved[0], resolved[1], resolved[2]))
                task_splits.append(partition)
                mapping_sources["api"] += 1

        note(
            f"Mapped {len(tasks)} / {len(proteins)} ProteinInvBench chains to AlphaFold "
            f"({mapping_sources['local']} local, {mapping_sources['api']} API)."
        )
        if not tasks:
            raise ValueError(
                "No ProteinInvBench chains could be mapped to AlphaFold structures. "
                f"Indexed {len(alphafold_index)} local AlphaFold files at {af_path}. "
                "Check that D_AFSP00 raw data is present and SIFTS mapping is available."
            )

        def process_protein(args):
            item, uniprot, structure_path, _source = args
            record = PdbProcessor.process_file(structure_path)
            data = {field: record[field] for field in record.fields}
            data["molecule_id"] = ak.Array([[item["name"]]])
            data["molecule_cath"] = ak.Array([[item["CATH"]]])
            data["molecule_uniprot"] = ak.Array([[uniprot]])
            data["molecule_structure_source"] = ak.Array([["alphafold"]])
            return ak.Record(data)

        records = list(
            parallelize(
                process_protein,
                tasks,
                description="Processing AlphaFold structures",
            )
        )
        split = np.array(task_splits)

        sort_index = np.argsort(split)
        records = [records[index] for index in sort_index]
        split = split[sort_index]

        assets = Assets(
            {
                "mapping_summary": {
                    "total": int(len(proteins)),
                    "mapped": len(records),
                    "mapped_local": mapping_sources["local"],
                    "mapped_api": mapping_sources["api"],
                    "skipped_no_uniprot": len(skipped["no_uniprot"]),
                    "skipped_no_alphafold": len(skipped["no_alphafold"]),
                    "af_name": af_name,
                    "af_version": af_version,
                    "use_api_fallback": use_api_fallback,
                    "api_workers": api_workers,
                },
                "skipped": skipped,
            }
        )

        return (
            batched(IteratorWithLength(iter(records), len(records))),
            Split(
                {"ProteinInvBench_scene_split": split},
                default="ProteinInvBench_scene_split",
            ),
            assets,
        )
