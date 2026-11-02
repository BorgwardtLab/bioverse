import awkward as ak

from ..metric import Metric


class MeanCatalyticRankMetric(Metric):
    """Mean 1-based rank of catalytic residues by predicted attribution score.

    Operates along the last axis so the parent :class:`~bioverse.metric.Metric`
    ``on_axis`` / ``per`` flattening can keep one protein (or batch of proteins)
    together. Defaults match residue-level metrics such as Recovery:
    flatten the batch axis (``on_axis=2``) and score each update separately
    (``per=1``).
    """

    better = "lower"

    def __init__(self, name="Mean Catalytic Rank", on_axis=2, per=1, **kwargs):
        super().__init__(name=name, on_axis=on_axis, per=per, **kwargs)

    def compute(self, y_true, y_pred):
        y_true = ak.values_astype(y_true, bool)
        order = ak.argsort(y_pred, axis=-1, ascending=False)
        sorted_true = y_true[order]
        ranks = ak.local_index(sorted_true, axis=-1) + 1
        mean_rank = ak.mean(ranks[sorted_true], axis=-1)
        n_pos = ak.sum(sorted_true, axis=-1)
        return ak.fill_none(mean_rank, 0.0) * (n_pos > 0)
