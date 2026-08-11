import awkward as ak
import numpy as np

from ..metric import Metric


class MeanAbsoluteErrorMetric(Metric):
    """Mean absolute error between predictions and targets."""

    better = "lower"

    def __init__(self, name="MAE", **kwargs):
        super().__init__(name=name, **kwargs)

    def compute(self, y_true, y_pred):
        mae = ak.mean(np.abs(y_true - y_pred), axis=-1)
        return mae
