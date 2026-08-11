from itertools import chain
from pathlib import Path

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
    extract,
    unzip_file,
)

# Test subsample per RNA-Puzzles case: 1000 total, targeting 100 near-natives
# and 900 decoys. When fewer near-natives exist, decoys fill to 1000.
TEST_EXAMPLES_PER_PUZZLE = 1000
TEST_NEAR_NATIVES_PER_PUZZLE = 100


def _sample_pdb_files(
    pdbs: list[Path], n: int, rng: np.random.Generator
) -> list[Path]:
    if len(pdbs) <= n:
        return pdbs
    indices = np.sort(rng.choice(len(pdbs), size=n, replace=False))
    return [pdbs[i] for i in indices]


def _select_test_pdb_files(test_path: Path) -> list[Path]:
    """Return test PDB paths: up to 100 near-natives and 900 decoys per puzzle."""
    rng = np.random.default_rng(config.seed)
    files: list[Path] = []

    decoys_root = test_path / "decoys" / "decoys"
    near_natives_root = test_path / "near_natives"
    for puzzle_dir in sorted(decoys_root.iterdir()):
        if not puzzle_dir.is_dir():
            continue

        near_native_dir = near_natives_root / f"{puzzle_dir.name}_near_native"
        near_native_pdbs = (
            sorted(near_native_dir.rglob("*.pdb"))
            if near_native_dir.is_dir()
            else []
        )
        n_near_native = min(
            TEST_NEAR_NATIVES_PER_PUZZLE, len(near_native_pdbs)
        )
        n_decoy = TEST_EXAMPLES_PER_PUZZLE - n_near_native

        decoy_pdbs = sorted(puzzle_dir.glob("*.pdb"))
        if len(decoy_pdbs) < n_decoy:
            raise ValueError(
                f"{puzzle_dir.name} has {len(decoy_pdbs)} decoys, "
                f"need {n_decoy} to reach {TEST_EXAMPLES_PER_PUZZLE} examples "
                f"with {n_near_native} near-natives."
            )

        files.extend(_sample_pdb_files(near_native_pdbs, n_near_native, rng))
        files.extend(_sample_pdb_files(decoy_pdbs, n_decoy, rng))

    return files


class AresAdapter(Adapter):
    """Download protein-ligand complexes from the ARES docking benchmark."""

    @classmethod
    def download(cls):
        path = config.raw_path / "ARES"
        base_url = "https://stacks.stanford.edu/file/bn398fc4306/"
        train_val_path = path / "classics_train_val" / "classics_train_val"
        test_path = path / "augmented_puzzles" / "augmented_puzzles"
        download(f"{base_url}/classics_train_val.tar", path / "classics_train_val")
        download(f"{base_url}/augmented_puzzles.tar", path / "augmented_puzzles")
        extract(path / "augmented_puzzles" / "augmented_puzzles" / "decoys.tar")
        for puzzle in (
            path / "augmented_puzzles" / "augmented_puzzles" / "near_natives"
        ).glob("*.tar.gz"):
            extract(unzip_file(puzzle))
        train = AresPdbProcessor.process(train_val_path / "example_train")
        val = AresPdbProcessor.process(train_val_path / "example_val")
        test = AresPdbProcessor.process(_select_test_pdb_files(test_path))
        n_train, n_val, n_test = len(train), len(val), len(test)
        batches = IteratorWithLength(chain(train, val, test), n_train + n_val + n_test)
        split = Split(
            {
                "ares_scene_split": ["train"] * n_train
                + ["val"] * n_val
                + ["test"] * n_test
            },
            default="ares_scene_split",
        )
        return batched(batches), split, Assets({})


# override PdbProcessor to add molecule_rms
class AresPdbProcessor(PdbProcessor):
    """Parse ARES benchmark PDB files with dataset-specific conventions."""

    @classmethod
    def process_file(cls, path: str | Path) -> ak.Record:
        record = super().process_file(path)
        with open(path, "r") as file:
            props = file.read().split("TER")[-1].split()
            rms = ak.Array([[float(dict(zip(props[0::2], props[1::2]))["rms"])]])
            record = ak.Record(
                {
                    **{k: record[k] for k in record.fields if k != "molecule_rms"},
                    "molecule_rms": rms,
                }
            )
        return record
