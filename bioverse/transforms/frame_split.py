from ..data import Split
from ..transform import Transform
from ..utilities import config


class FrameSplit(Transform):

    def __init__(self, train_size=0.8, test_size=0.1, val_size=0.1) -> None:
        super().__init__()
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size

    def transform(self, batches, split, assets):
        n = int(sum(len(d) for d in batches.copy()))
        n_test = (
            int(n * self.test_size)
            if isinstance(self.test_size, float)
            else self.test_size
        )
        n_val = (
            int(n * self.val_size)
            if isinstance(self.val_size, float)
            else self.val_size
        )
        n_train = n - n_test - n_val
        index = [[0]] * n_train + [[1]] * n_test + [[2]] * n_val
        split = Split(index, names=["train", "test", "val"])
        return batches, split, assets
