import awkward as ak

from ..task import Task


class PropertyPredictionTask(Task):

    def __init__(self, property="label", level="molecule", resolution="atom") -> None:
        super().__init__()
        self.property = property
        self.level = level
        self.resolution = resolution

    def __call__(self, vbatch, assets, index):
        X = vbatch[index["scene"], index["frame"], index["molecule"]]
        X.resolution = self.resolution
        if isinstance(self.property, tuple) or isinstance(self.property, list):
            prop, prop_idx = self.property
            targets = X.molecules.__getattr__(f"{self.level}_{prop}")[:, prop_idx]
        else:
            targets = X.molecules.__getattr__(f"{self.level}_{self.property}")
        y = ak.Array({"target": targets})
        return X, y
