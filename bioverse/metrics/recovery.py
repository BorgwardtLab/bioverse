from .multi_class_accuracy import MultiClassAccuracyMetric


class RecoveryMetric(MultiClassAccuracyMetric):
    better = "higher"

    def __init__(self, name="Recovery", **kwargs):
        super().__init__(name=name, **kwargs)
