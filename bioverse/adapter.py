from abc import ABC, abstractmethod
from typing import Iterator, Tuple

from bioverse.data import Assets, Batch, Split


class Adapter(ABC):

    @abstractmethod
    def download(self, *args, **kwargs) -> Tuple[Iterator[Batch], Split, Assets]:
        """
        Implement this method to download raw data from a repository and return the `Batch`-iterator, `Split` and `Assets` objects.
        """
        pass
