import awkward as ak

from ..metric import Metric


class MeanSquaredErrorMetric(Metric):
    better = "lower"

    def __init__(self, name="MSE", **kwargs):
        super().__init__(name=name, **kwargs)

    def compute(self, y_true, y_pred):
        mse = ak.mean((y_true - y_pred) ** 2, axis=-1)
        return mse
