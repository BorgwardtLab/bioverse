import awkward as ak
import numpy as np

from ..metric import Metric


class MultiClassAccuracyMetric(Metric):
    """Multi-class classification accuracy (fraction of argmax-correct predictions).

    Base class for metrics that compare predicted and true class indices via
    ``argmax``. Subclassed by :class:`~bioverse.metrics.recovery.RecoveryMetric`.
    """

    better = "higher"

    def __init__(self, name="Accuracy", **kwargs):
        super().__init__(name=name, **kwargs)

    def compute(self, y_true, y_pred):
        y_true = ak.to_numpy(ak.ravel(y_true))
        if y_pred.ndim > 1:
            y_pred = ak.argmax(y_pred, axis=-1)
        y_pred = ak.to_numpy(ak.ravel(y_pred))
        return float(np.mean(y_true == y_pred))
