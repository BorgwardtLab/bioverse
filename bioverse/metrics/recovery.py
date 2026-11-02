from .multi_class_accuracy import MultiClassAccuracyMetric


class RecoveryMetric(MultiClassAccuracyMetric):
    """Sequence recovery rate (fraction of correctly predicted residues)."""

    better = "higher"

    def __init__(self, name="Recovery", on_axis=2, per=1, reduction="nanmean", **kwargs):
        super().__init__(name=name, on_axis=on_axis, per=per, reduction=reduction, **kwargs)
