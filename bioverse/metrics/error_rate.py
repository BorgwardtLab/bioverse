import awkward as ak

from ..metric import Metric


class ErrorRateMetric(Metric):
    better = "lower"

    def __init__(self, name="Error Rate", **kwargs):
        super().__init__(name=name, **kwargs)

    def compute(self, y_true, y_pred):
        y_pred = ak.argmax(y_pred, axis=-1)
        correct = ak.sum(y_true == y_pred, axis=-1)
        total = ak.num(y_true, axis=-1)
        return (1 - (correct / total)) * 100
