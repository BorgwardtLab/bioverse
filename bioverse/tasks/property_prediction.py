import awkward as ak

from ..task import Task


class PropertyPredictionTask(Task):

    def __init__(self, property="label", resolution="atom") -> None:
        super().__init__()
        self.property = property
        self.resolution = resolution

    def __call__(self, vbatch, assets, index):
        X = vbatch[index["scene"], index["frame"], index["molecule"]]
        X.resolution = self.resolution
        targets = X.molecules.__getattr__(f"molecule_{self.property}")
        y = ak.Array({"target": targets})
        y.attrs["level"] = "molecule"
        return X, y
