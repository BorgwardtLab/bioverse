from itertools import chain
from pathlib import Path

import awkward as ak

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


class AresAdapter(Adapter):
    """Adapter for ARES. Paper: https://www.science.org/doi/10.1126/science.abe5650"""

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
        test = AresPdbProcessor.process(test_path)
        n_train, n_val, n_test = len(train), len(val), len(test)
        batches = IteratorWithLength(chain(train, val, test), n_train + n_val + n_test)
        split = Split(
            [[0]] * n_train + [[1]] * n_val + [[2]] * n_test, ["train", "val", "test"]
        )
        return batched(batches), split, Assets({})


# override PdbProcessor to add molecule_rms
class AresPdbProcessor(PdbProcessor):
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
