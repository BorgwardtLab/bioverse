from ..task import Task


class VertexPropertyPredictionTask(Task):

    def __init__(self, property=None) -> None:
        super().__init__()
        self.property = property if isinstance(property, list) else [property]

    def __call__(self, dataset, index):
        X = dataset[index["scene"], index["frame"], index["molecule"]]
        y = X[self.property][index["vertex"]]
        y["__index__"] = index["vertex"]
        return X, y
