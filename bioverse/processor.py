import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List

import awkward as ak
import numpy as np

from .utilities import IteratorWithLength, config, note, parallelize


class Processor(ABC):
    """Parse raw structure files into Awkward Array records.

    Processors convert files on disk (PDB, mmCIF, SDF, …) into uniform
    ``ak.Record`` objects that adapters can shard into
    :class:`~bioverse.data.Batch` instances. Subclasses declare supported
    extensions via :attr:`valid_extensions` and implement
    :meth:`process_file`.

    Call :meth:`process` to walk a directory tree and parse all matching
    files in parallel.

    Examples
    --------
    .. code-block:: python

       from bioverse.processors.pdb import PdbProcessor

       records = PdbProcessor.process("/data/pdb_structures")
       for record in records:
           print(record.scene_id, len(record.residue_token))
    """

    valid_extensions: List[str] = []

    @classmethod
    def process(
        cls,
        path: Path | str | list[Path | str],
        shuffle: bool = True,
        exclude: Iterable[str] | None = None,
    ) -> IteratorWithLength[ak.Record]:
        note("Building file tree for processing.")
        exclude_set = set(exclude) if exclude is not None else None
        if not isinstance(path, list):
            path = Path(path)
            files = []
            for root, _, names in os.walk(path):
                for name in names:
                    file_path = Path(root) / name
                    suffix = "".join(file_path.suffixes)
                    if suffix in cls.valid_extensions and not cls._is_excluded(
                        file_path, exclude_set
                    ):
                        files.append(file_path)
        else:
            files = [
                Path(p)
                for p in path
                if not cls._is_excluded(Path(p), exclude_set)
            ]
        files = sorted(files)
        if shuffle:
            np.random.default_rng(config.seed).shuffle(files)
        processed = parallelize(
            cls.process_file,
            files,
            description="Processing",
        )
        return IteratorWithLength(filter(lambda x: x is not None, processed))

    @classmethod
    def _is_excluded(
        cls, path: Path, exclude: set[str] | None
    ) -> bool:
        if exclude is None:
            return False
        key = cls.exclude_key(path)
        return key is not None and key in exclude

    @classmethod
    def exclude_key(cls, path: Path | str) -> str | None:
        return None

    @classmethod
    @abstractmethod
    def process_file(cls, path: Path | str) -> ak.Record | None:
        raise NotImplementedError
