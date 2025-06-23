import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import awkward as ak
import numpy as np

from .utilities import IteratorWithLength, config, note, parallelize


class Processor(ABC):

    valid_extensions: List[str] = []

    @classmethod
    def process(
        cls, path: Path | str | list[Path | str], shuffle: bool = True
    ) -> IteratorWithLength[ak.Record]:
        note("Building file tree for processing.")
        if not isinstance(path, list):
            path = Path(path)
            files = []
            for root, _, names in os.walk(path):
                for name in names:
                    file_path = Path(root) / name
                    suffix = "".join(file_path.suffixes)
                    if suffix in cls.valid_extensions:
                        files.append(file_path)
        else:
            files = [Path(p) for p in path]
        files = sorted(files)
        if shuffle:
            np.random.default_rng(config.seed).shuffle(files)
        processed = parallelize(
            cls.process_file,
            files,
            description="Processing",
            max_workers=20,
        )
        return IteratorWithLength(filter(lambda x: x is not None, processed))

    @classmethod
    @abstractmethod
    def process_file(cls, path: Path | str) -> ak.Record | None:
        raise NotImplementedError
