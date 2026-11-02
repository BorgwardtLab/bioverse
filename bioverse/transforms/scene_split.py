import numpy as np

from ..data import Split
from ..transform import Transform


def stratified_split_index(
    labels,
    train_size=0.8,
    test_size=0.1,
    val_size=0.1,
    seed=0,
):
    """Assign shuffled, class-balanced train/test/val labels.

    Each class is shuffled independently, then split in proportion to
    ``train_size`` / ``test_size`` / ``val_size``. Remainder scenes fill
    leftover quota so the partition sizes stay exact. A dataset with a
    single class (or no labels) is shuffled as one group.
    """
    labels = np.asarray(labels)
    n = int(labels.size)
    if n == 0:
        return []

    n_test = int(n * test_size) if isinstance(test_size, float) else int(test_size)
    n_val = int(n * val_size) if isinstance(val_size, float) else int(val_size)
    n_test = max(0, min(n_test, n))
    n_val = max(0, min(n_val, n - n_test))
    n_train = n - n_test - n_val

    rng = np.random.default_rng(seed)
    index = np.empty(n, dtype=object)
    quotas = {"train": n_train, "test": n_test, "val": n_val}
    assigned = {name: 0 for name in quotas}
    leftover = []

    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        n_c = int(idx.size)
        n_c_train = (n_c * n_train) // n
        n_c_test = (n_c * n_test) // n
        n_c_val = (n_c * n_val) // n
        used = n_c_train + n_c_test + n_c_val
        index[idx[:n_c_train]] = "train"
        index[idx[n_c_train : n_c_train + n_c_test]] = "test"
        index[idx[n_c_train + n_c_test : used]] = "val"
        assigned["train"] += n_c_train
        assigned["test"] += n_c_test
        assigned["val"] += n_c_val
        leftover.extend(idx[used:].tolist())

    leftover = np.asarray(leftover, dtype=int)
    rng.shuffle(leftover)
    offset = 0
    for name in ("train", "test", "val"):
        need = quotas[name] - assigned[name]
        if need > 0:
            take = leftover[offset : offset + need]
            index[take] = name
            offset += need

    if np.any(index == None):  # noqa: E711
        raise RuntimeError("stratified split left unassigned scenes")
    return index.tolist()


def scene_class_labels(batches):
    """One class label per scene, or ``-1`` when no label field exists."""
    labels = []
    for batch in batches:
        n = len(batch)
        key = next(
            (
                name
                for name in ("molecule_present", "molecule_label", "molecule_y")
                if name in batch
            ),
            None,
        )
        if key is None:
            labels.append(np.full(n, -1, dtype=np.int64))
            continue
        values = np.asarray(ak_ravel(getattr(batch, key)))
        if values.size < n:
            values = np.pad(values, (0, n - values.size), constant_values=-1)
        labels.append(values.reshape(-1)[:n].astype(np.int64))
    if not labels:
        return np.array([], dtype=np.int64)
    return np.concatenate(labels, axis=0)


def ak_ravel(array):
    import awkward as ak

    return ak.to_numpy(ak.ravel(array))


class SceneSplit(Transform):

    """Assign shuffled, class-balanced split partitions at the scene level."""

    def __init__(self, train_size=0.8, test_size=0.1, val_size=0.1, seed=0) -> None:
        super().__init__()
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.seed = seed

    def transform(self, batches, split, assets):
        labels = scene_class_labels(batches.copy())
        index = stratified_split_index(
            labels,
            train_size=self.train_size,
            test_size=self.test_size,
            val_size=self.val_size,
            seed=self.seed,
        )
        split = Split({"random_scene_split": index}, default="random_scene_split")
        return batches, split, assets
