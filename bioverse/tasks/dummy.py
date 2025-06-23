from ..task import Task
import awkward as ak


class DummyTask(Task):

    def __call__(self, dataset, index):
        X = dataset[index["scene"], index["frame"], index["molecule"]]
        y = X[["tokens"]]
        y["__index__"] = None
        return X, y
