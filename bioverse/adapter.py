from abc import ABC, abstractmethod
from typing import Iterator, Tuple

from bioverse.data import Assets, Batch, Split


class Adapter(ABC):
    """Download raw data from an external source into Bioverse.

    Adapters are the entry point of the data pipeline. A concrete adapter
    fetches or generates files under :data:`~bioverse.utilities.config.raw_path`,
    optionally parses them with a :class:`~bioverse.processor.Processor`, and
    returns the three objects required to build a
    :class:`~bioverse.dataset.Dataset`:

    * an iterator of :class:`~bioverse.data.Batch` shards,
    * a :class:`~bioverse.data.Split` with partition assignments, and
    * :class:`~bioverse.data.Assets` for shared lookup tables.

    Implement :meth:`download` and register the class in a dataset config
    (``D_*.yaml``) or call it from :meth:`~bioverse.dataset.Dataset.release`.

    Examples
    --------
    Minimal adapter that wraps a local PDB directory:

    .. code-block:: python

       from bioverse.adapter import Adapter
       from bioverse.data import Assets, Batch, Split
       from bioverse.processors.pdb import PdbProcessor
       from bioverse.utilities.io import rebatch

       class MyAdapter(Adapter):
           def download(self, path="my_structures"):
               records = PdbProcessor.process(path)
               batches = rebatch(records)
               split = Split({"scene_split": ["train"] * len(batches)})
               assets = Assets()
               return batches, split, assets
    """

    @abstractmethod
    def download(self, *args, **kwargs) -> Tuple[Iterator[Batch], Split, Assets]:
        """Download raw data and return batches, split, and assets.

        Returns
        -------
        batches
            Iterator yielding :class:`~bioverse.data.Batch` shards.
        split
            Train/validation/test partition assignments.
        assets
            Auxiliary lookup tables shared across batches.
        """
        pass
