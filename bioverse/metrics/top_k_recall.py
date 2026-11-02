import awkward as ak

from ..metric import Metric


class TopKRecallMetric(Metric):
    """Fraction of catalytic residues recovered in the top-k attribution scores.

    Rankings are computed along the last axis. Defaults match residue-level
    metrics such as Recovery (``on_axis=2``, ``per=1``).
    """

    better = "higher"

    def __init__(self, name="Top-k Recall", k: int = 5, on_axis=2, per=1, **kwargs):
        if k < 1:
            raise ValueError("TopKRecallMetric requires k >= 1.")
        super().__init__(name=f"Top-{k} Recall", on_axis=on_axis, per=per, **kwargs)
        self.k = int(k)

    def compute(self, y_true, y_pred):
        y_true = ak.values_astype(y_true, bool)
        order = ak.argsort(y_pred, axis=-1, ascending=False)
        sorted_true = y_true[order]
        hits = ak.sum(sorted_true[..., : self.k], axis=-1)
        n_pos = ak.sum(sorted_true, axis=-1)
        return ak.fill_none(hits / n_pos, 0.0) * (n_pos > 0)
