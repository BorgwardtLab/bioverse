import awkward as ak
import numpy as np

from ..transform import Transform


class Standardize(Transform):

    def __init__(self, field):
        self.field = field

    def fit(self, batches, split, assets):
        values = []
        for batch in batches:
            values.append(batch.__getattr__(self.field))
        values = ak.concatenate(values, axis=0)
        self.mean = ak.mean(values, axis=0)
        self.std = ak.std(values, axis=0)
        if values.ndim > 1:
            self.mean = np.array(self.mean).reshape(1, -1)
            self.std = np.array(self.std).reshape(1, -1)

    def transform_batch(self, batch):
        batch.__setattr__(
            self.field, (batch.__getattr__(self.field) - self.mean) / self.std
        )
        return batch

    def inverse_transform(self, y):
        y["target"] = y["target"] * self.std + self.mean
        return y
