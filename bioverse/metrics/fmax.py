import numpy as np

from ..metric import Metric


class FmaxMetric(Metric):
    better = "higher"

    def __init__(self, name="Fmax", **kwargs):
        super().__init__(name=name, **kwargs)

    def compute(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        fmax = 0.0
        for threshold in np.linspace(0, 1, 21):
            pred = y_pred >= threshold
            tp = np.logical_and(y_true, pred).sum(axis=-1)
            fp = np.logical_and(~y_true, pred).sum(axis=-1)
            fn = np.logical_and(y_true, ~pred).sum(axis=-1)
            precision = np.divide(
                tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0
            )
            recall = np.divide(
                tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0
            )
            f1 = np.divide(
                2 * precision * recall,
                precision + recall,
                out=np.zeros_like(precision, dtype=float),
                where=(precision + recall) != 0,
            )
            fmax = max(fmax, float(np.mean(f1)))
        return fmax
