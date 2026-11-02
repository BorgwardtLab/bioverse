import awkward as ak
import numpy as np

from ..metric import Metric


class TopKEnrichmentMetric(Metric):
    """Enrichment of beneficial labels among the model's top-k predictions.

    A label is beneficial when it is strictly above ``threshold``, or above
    the mean of ``y_true`` when ``threshold`` is ``None``. The score is
    ``(beneficial in top-k / k) / (beneficial / n)``, i.e. 1.0 is random.
    """

    better = "higher"

    def __init__(self, name="Top-k enrichment", k: int = 10, threshold=None, **kwargs):
        if k < 1:
            raise ValueError("TopKEnrichmentMetric requires k >= 1.")
        super().__init__(name=f"Top-{k} enrichment", **kwargs)
        self.k = int(k)
        self.threshold = threshold

    def compute(self, y_true, y_pred):
        true = np.asarray(ak.to_numpy(ak.ravel(y_true)), dtype=np.float64).reshape(-1)
        pred = np.asarray(ak.to_numpy(ak.ravel(y_pred)), dtype=np.float64).reshape(-1)
        n = true.size
        if n == 0:
            return float("nan")
        cutoff = float(true.mean()) if self.threshold is None else float(self.threshold)
        beneficial = true > cutoff
        n_pos = int(beneficial.sum())
        if n_pos == 0:
            return float("nan")
        k = min(self.k, n)
        order = np.argsort(pred, kind="mergesort")[::-1]
        hits = int(beneficial[order[:k]].sum())
        return float((hits / k) / (n_pos / n))
