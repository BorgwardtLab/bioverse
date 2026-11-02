import awkward as ak

from ..metric import Metric


class TopKPrecisionMetric(Metric):
    """Precision of the top-k attribution scores against catalytic residues.

    Rankings are computed along the last axis. Defaults match residue-level
    metrics such as Recovery (``on_axis=2``, ``per=1``).
    """

    better = "higher"

    def __init__(self, name="Top-k Precision", k: int = 5, on_axis=2, per=1, **kwargs):
        if k < 1:
            raise ValueError("TopKPrecisionMetric requires k >= 1.")
        super().__init__(name=f"Top-{k} Precision", on_axis=on_axis, per=per, **kwargs)
        self.k = int(k)

    def compute(self, y_true, y_pred):
        y_true = ak.values_astype(y_true, bool)
        order = ak.argsort(y_pred, axis=-1, ascending=False)
        sorted_true = y_true[order]
        topk = sorted_true[..., : self.k]
        hits = ak.sum(topk, axis=-1)
        k_used = ak.num(topk, axis=-1)
        n_pos = ak.sum(sorted_true, axis=-1)
        return ak.fill_none(hits / k_used, 0.0) * (n_pos > 0) * (k_used > 0)
