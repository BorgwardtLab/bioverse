import struct
import zlib

import awkward as ak
import numpy as np
import pandas as pd

from ..adapter import Adapter
from ..data import Assets, Split
from ..utilities import IteratorWithLength, batched, config, download


class RotMnistAdapter(Adapter):
    """Adapter for rotated MNIST."""

    # from https://github.com/tscohen/gconv_experiments

    @classmethod
    def download(cls):
        path = config.raw_path / "ROT_MNIST"
        download(
            "http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip",
            path,
        )
        train = np.loadtxt(
            path / "mnist_all_rotation_normalized_float_train_valid.amat"
        )
        test = np.loadtxt(path / "mnist_all_rotation_normalized_float_test.amat")

        labels = np.concatenate([train[:, -1], test[:, -1]]).astype(np.int32)
        train_data = train[:, :-1].reshape(-1, 28, 28)
        test_data = test[:, :-1].reshape(-1, 28, 28)
        data = np.concatenate([train_data, test_data])

        # compute split
        n_train, n_test = int(len(train_data) * 0.8), len(test_data)
        n_val = len(train_data) - n_train
        split = np.array([[0]] * n_train + [[1]] * n_val + [[2]] * n_test)

        # convert to point cloud
        coords = [np.argwhere(item) for item in data]
        values = [item[xy[:, 0], xy[:, 1]] for item, xy in zip(data, coords)]

        # create records
        def generator():
            for coord, value, label in zip(coords, values, labels):
                coord = coord.astype(np.float32)
                value = value.astype(np.float32)[..., None]
                data = {
                    "molecule_label": [[label]],
                    "atom_pos": [[[[coord]]]],
                    "atom_features": [[[[value]]]],
                }
                yield ak.Record(data)

        batches = batched(IteratorWithLength(generator(), len(coords)))
        return batches, Split(split, names=["train", "val", "test"]), Assets({})
