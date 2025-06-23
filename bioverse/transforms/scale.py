import awkward as ak
import numpy as np

from ..transform import Transform


class Scale(Transform):

    def __init__(self, field, lo=None, hi=None):
        self.field = field
        self.lo = lo
        self.hi = hi

    def fit(self, batches, split, assets):
        if not self.lo and not self.hi:
            self.lo, self.hi = np.inf, -np.inf
            for batch in batches:
                lo = ak.min(batch.__getattr__(self.field), axis=-1)
                hi = ak.max(batch.__getattr__(self.field), axis=-1)
                self.lo = ak.min([self.lo, lo])
                self.hi = ak.max([self.hi, hi])

    def transform_batch(self, batch):
        batch.__setattr__(
            self.field, (batch.__getattr__(self.field) - self.lo) / (self.hi - self.lo)
        )
        return batch

    def inverse_transform(self, y):
        y["target"] = y["target"] * (self.hi - self.lo) + self.lo
        return y
