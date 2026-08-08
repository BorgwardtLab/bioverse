import numpy as np

from ..transform import Transform


class StandardizePairwise(Transform):
    """Standardize pairwise targets stored in assets (e.g. lDDT matrices)."""

    def __init__(self, meta: str = "lddt"):
        self.meta = meta

    def fit(self, batches, split, assets):
        table = assets[self.meta]
        matrix = np.load(table["path"], mmap_mode="r").astype(np.float64)
        self.mean = float(np.mean(matrix))
        self.std = float(np.std(matrix))
        if self.std == 0:
            self.std = 1.0

    def transform_assets(self, assets):
        table = dict(assets[self.meta])
        table["standardize_mean"] = float(self.mean)
        table["standardize_std"] = float(self.std)
        assets[self.meta] = table
        return assets

    def inverse_transform(self, y):
        y["target"] = y["target"] * self.std + self.mean
        return y
