import numpy as np
import awkward as ak

from ..metric import Metric


class MacroRecallMetric(Metric):
    """Macro-averaged recall over classes."""

    better = "higher"

    def __init__(self, name="Recall", **kwargs):
        super().__init__(name=name, **kwargs)

    def compute(self, y_true, y_pred):
        y_true = np.asarray(ak.ravel(y_true))
        y_pred = np.asarray(ak.argmax(y_pred, axis=-1) if y_pred.ndim > 1 else y_pred)
        classes = np.unique(y_true)
        scores = []
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            scores.append(tp / (tp + fn) if (tp + fn) else 0.0)
        return float(np.mean(scores))
