import awkward as ak
import numpy as np

from ..metric import Metric
from ..utilities import BLOSUM62


class BlosumScoreMetric(Metric):
    """Average BLOSUM62 substitution score between true and predicted residues."""

    better = "higher"

    def __init__(self, name: str = "Blosum Score", on_axis: int = 2, per: int = 1, reduction="nanmean", **kwargs):
        super().__init__(name=name, on_axis=on_axis, per=per, reduction=reduction, **kwargs)

    def compute(self, y_true: ak.Array, y_pred: ak.Array):
        if y_pred.ndim > y_true.ndim:
            y_pred = ak.argmax(y_pred, axis=-1)

        lengths = np.asarray(ak.to_numpy(ak.num(y_true, axis=-1)), dtype=np.int64)
        true_flat = ak.to_numpy(ak.flatten(y_true, axis=-1))
        pred_flat = ak.to_numpy(ak.flatten(y_pred, axis=-1))
        if true_flat.size == 0 or pred_flat.size == 0:
            return np.full(max(int(lengths.size), 1), np.nan)
        scores_flat = BLOSUM62[true_flat, pred_flat]
        out = np.empty(lengths.size, dtype=np.float64)
        start = 0
        for i, n in enumerate(lengths.ravel()):
            n = int(n)
            if n == 0:
                out.ravel()[i] = np.nan
            else:
                out.ravel()[i] = float(np.mean(scores_flat[start : start + n]))
            start += n
        return out if out.ndim > 0 else float(out)
