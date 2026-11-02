import os
from pathlib import Path

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
    parallelize,
)
from ..utilities.id_mapping import load_invbench_test_domain_uniprots

EXCLUSIONS_PATH = Path(__file__).parent / "data" / "foldseek_exclusions.json"
EXCLUDED_UNIPROTS = frozenset(load(EXCLUSIONS_PATH)["excluded_uniprots"])


class AlphaFoldExclusionAdapter(Adapter):
    """Download AlphaFold structures with homology and CATH-domain test exclusions."""

    @classmethod
    def download(
        cls,
        name: str = "swissprot_pdb",
        version: str = "v4",
        test_size: float | int = 1000,
        val_size: float | int = 1000,
        split_seed: int = 0,
        limit: int | None = None,
    ):
        path = config.raw_path / "AlphaFoldDB" / version / name
        base_url = "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/"
        download(f"{base_url}/{name}_{version}.tar", path)
        glob_delete(str(path / "*.cif.gz"))
        test_domain_uniprots = frozenset(load_invbench_test_domain_uniprots())
        excluded = EXCLUDED_UNIPROTS | test_domain_uniprots

        files = []
        for root, _, names in os.walk(path):
            for filename in names:
                file_path = Path(root) / filename
                suffix = "".join(file_path.suffixes)
                if suffix not in PdbProcessor.valid_extensions:
                    continue
                if PdbProcessor._is_excluded(file_path, excluded):
                    continue
                files.append(file_path)
        files = sorted(files)
        if limit is not None:
            files = files[:limit]

        n_total = len(files)
        n_test = int(n_total * test_size) if isinstance(test_size, float) else int(test_size)
        n_val = int(n_total * val_size) if isinstance(val_size, float) else int(val_size)
        n_test = max(0, min(n_test, n_total))
        n_val = max(0, min(n_val, n_total - n_test))
        n_train = n_total - n_test - n_val

        rng = np.random.default_rng(split_seed)
        order = rng.permutation(n_total)
        train_idx = set(order[:n_train].tolist())
        val_idx = set(order[n_train : n_train + n_val].tolist())
        test_idx = set(order[n_train + n_val :].tolist())
        by_partition = {"train": [], "val": [], "test": []}
        for i, file_path in enumerate(files):
            if i in train_idx:
                by_partition["train"].append(file_path)
            elif i in val_idx:
                by_partition["val"].append(file_path)
            elif i in test_idx:
                by_partition["test"].append(file_path)

        ordered_files = []
        split_labels = []
        for partition in ("train", "val", "test"):
            bucket = by_partition[partition]
            if bucket:
                order = rng.permutation(len(bucket))
                bucket = [bucket[int(i)] for i in order]
            ordered_files.extend(bucket)
            split_labels.extend([partition] * len(bucket))

        raw_records = parallelize(
            PdbProcessor.process_file,
            ordered_files,
            description="Processing structures",
        )
        records = []
        kept_splits = []
        for record, label in zip(raw_records, split_labels):
            if record is None:
                continue
            records.append(record)
            kept_splits.append(label)

        batches = batched(IteratorWithLength(iter(records), len(records)))
        assets = Assets(
            {
                "exclusions": {
                    "path": str(EXCLUSIONS_PATH),
                    "count": len(excluded),
                    "foldseek_count": len(EXCLUDED_UNIPROTS),
                    "test_domain_count": len(test_domain_uniprots),
                    "split_seed": split_seed,
                    "split_sizes": {
                        partition: kept_splits.count(partition)
                        for partition in ("train", "val", "test")
                    },
                }
            }
        )
        return (
            batches,
            Split({"alphafold_scene_split": kept_splits}, default="alphafold_scene_split"),
            assets,
        )
