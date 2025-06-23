import awkward as ak

from ..transform import Transform


class Standardize(Transform):

    def __init__(self, field):
        self.field = field

    def fit(self, batches, split, assets):
        values = []
        for batch in batches:
            values.append(batch.__getattr__(self.field))
        values = ak.concatenate(values, axis=-1)
        self.mean = ak.mean(values, axis=-1)
        self.std = ak.std(values, axis=-1)

    def transform_batch(self, batch):
        batch.__setattr__(
            self.field, (batch.__getattr__(self.field) - self.mean) / self.std
        )
        return batch

    def inverse_transform(self, y):
        y["target"] = y["target"] * self.std + self.mean
        return y
