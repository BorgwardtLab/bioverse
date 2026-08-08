import awkward as ak
import numpy as np

from ..metric import Metric


class MultiClassAccuracyMetric(Metric):
    better = "higher"

    def __init__(self, name="Accuracy", **kwargs):
        super().__init__(name=name, **kwargs)

    def compute(self, y_true, y_pred):
        y_true = np.asarray(ak.ravel(y_true))
        y_pred = np.asarray(
            ak.argmax(y_pred, axis=-1) if y_pred.ndim > 1 else y_pred
        )
        return float(np.mean(y_true == y_pred))
