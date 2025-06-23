import itertools
import os
import shutil

from ..adapter import Adapter
from ..data import Assets, Split
from ..processors import CifProcessor
from ..utilities import IteratorWithLength, batched, config, download


class Rna3dbAdapter(Adapter):
    """Adapter for RNA3DB."""

    @classmethod
    def download(cls):
        path = config.raw_path / "RNA3DB"
        download(
            "https://github.com/marcellszi/rna3db/releases/download/2024-12-04-full-release/rna3db-mmcifs.tar.xz",
            path,
        )
        if not os.path.exists(path / "val_set"):
            os.makedirs(path / "val_set")
            for no in range(80, 100):
                if os.path.exists(path / f"train_set/component_{no}"):
                    shutil.move(
                        path / f"train_set/component_{no}",
                        path / f"val_set/component_{no}",
                    )
        train = CifProcessor.process(path / "train_set")
        val = CifProcessor.process(path / "val_set")
        test = CifProcessor.process(path / "test_set")
        batches = batched(
            IteratorWithLength(
                itertools.chain(train, val, test), len(train) + len(val) + len(test)
            )
        )
        split = [[0]] * len(train) + [[1]] * len(val) + [[2]] * len(test)
        return batches, Split(split, ["train", "val", "test"]), Assets({})
