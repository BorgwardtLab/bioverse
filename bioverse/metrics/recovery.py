from .multi_class_accuracy import MultiClassAccuracyMetric


class RecoveryMetric(MultiClassAccuracyMetric):
    better = "higher"

    def __init__(self, name="Recovery", on=2, per=1, **kwargs):
        super().__init__(name=name, on=on, per=per, **kwargs)
