import awkward as ak

from ..collater import Collater
from .long import LongCollater


class PairwiseData:
    def __init__(self, data1, data2, y=None, _sizes=None):
        self.data1 = data1
        self.data2 = data2
        self.y = y
        self._sizes = _sizes

    def uncollate(self, y):
        if self._sizes is not None:
            y = ak.unflatten(y, self._sizes, axis=0)
        return ak.Array({"target": y})


class PairwiseLongCollater(Collater):

    @classmethod
    def collate(cls, X, y=None, attr=[], assets=None):
        X1, X2 = X
        data1 = LongCollater.collate(X1, None, attr=attr, assets=assets)
        data2 = LongCollater.collate(X2, None, attr=attr, assets=assets)
        if y is None:
            return PairwiseData(data1, data2)
        y_arr = y["target"]
        if "sizes" in y.fields:
            y_arr = ak.flatten(y_arr, axis=1)
        return PairwiseData(
            data1,
            data2,
            y_arr,
            _sizes=(y["sizes"] if "sizes" in y.fields else None),
        )
