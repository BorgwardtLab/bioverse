import awkward as ak
import numpy as np

from ..metric import Metric


class PearsonsRMetric(Metric):
    better = "higher"

    def __init__(self, name="Pearson", **kwargs):
        super().__init__(name=name, **kwargs)

    def compute(self, y_true, y_pred):
        mean_y_true = ak.mean(y_true, axis=-1, keepdims=True)
        mean_y_pred = ak.mean(y_pred, axis=-1, keepdims=True)

        y_true_diff = y_true - mean_y_true
        y_pred_diff = y_pred - mean_y_pred

        numerator = ak.sum(y_true_diff * y_pred_diff, axis=-1)
        denominator = np.sqrt(
            ak.sum(y_true_diff**2, axis=-1) * ak.sum(y_pred_diff**2, axis=-1)
        )

        pearsons_r = numerator / denominator
        return pearsons_r
