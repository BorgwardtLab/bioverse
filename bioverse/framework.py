from abc import ABC, abstractmethod
from typing import Any, List

import awkward as ak

from .data import Batch


class Framework(ABC):

    @abstractmethod
    def collate(self, X: Batch, y: ak.Array | None = None, attr: List[str] = []) -> Any:
        pass
