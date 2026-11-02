import awkward as ak
import numpy as np

from ..metric import Metric


def pearson_r(left, right) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    mask = np.isfinite(left) & np.isfinite(right)
    left, right = left[mask], right[mask]
    if left.size < 2:
        return float("nan")
    left = left - left.mean()
    right = right - right.mean()
    denom = float(np.sqrt(np.sum(left * left) * np.sum(right * right)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(left * right) / denom)


class PearsonsRMetric(Metric):
    """Pearson correlation coefficient between predictions and targets."""

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
