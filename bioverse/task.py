from abc import ABC, abstractmethod
from typing import Tuple

import awkward as ak

from .data import Assets, Batch
from .virtual import VirtualBatch


class Task(ABC):

    @abstractmethod
    def __call__(
        self, vbatch: VirtualBatch, assets: Assets, index: Tuple[ak.Array, ...]
    ) -> Tuple[Batch, ak.Array]:

        raise NotImplementedError
