import awkward as ak
import numpy as np

from ..adapter import Adapter
from ..data import Assets, Split
from ..utilities import IteratorWithLength, batched, config


class PicassoAdapter(Adapter):
    """Adapter for the synthetic Picasso dataset. Used to demonstrate invariance and equivariance to rotation. Generates molecules looking like ∆---∆, with rotated triangles at the ends, with one relative orientation being the positive class and the remaining being the negative class."""

    @classmethod
    def download(cls):
        path = config.raw_path / "Picasso"

        # template triangle for the ends
        triangle = np.array([[0, 0], [-0.5, 1], [0.5, 1]])
        # create 12 regularly spaced rotation matrices
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        rotations = np.array(
            [
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ]
                for angle in angles
            ]
        )
        # sample all possible combinations of rotations for both ends
        top, bottom = np.repeat(np.arange(12), 12), np.tile(np.arange(12), 12)
        # rotate and translate ends
        top = np.einsum("ij,kjl->kil", triangle, rotations[top]) + [0, 2]
        bottom = np.einsum("ij,kjl->kil", triangle, rotations[bottom]) - [0, 2]
        # combine into molecule and add middle bar
        bar = np.array([[0, 1], [0, 0], [0, -1]]).reshape(1, 3, 2).repeat(144, axis=0)
        X_train = np.concatenate([top, bar, bottom], axis=1)
        pos = X_train[6].reshape(1, -1, 2).repeat(144, axis=0)
        X_train = np.concatenate([X_train, pos], axis=0)
        # assign labels
        y_train = np.concatenate([np.zeros(144), np.ones(144)])
        y_train[6] = 1  # triangles facing away
        # assign edges
        edge_index = (
            np.array(
                [
                    [0, 1],
                    [1, 2],
                    [2, 0],
                    [0, 3],
                    [3, 4],
                    [4, 5],
                    [5, 6],
                    [6, 7],
                    [7, 8],
                    [8, 6],
                    [1, 0],
                    [2, 1],
                    [0, 2],
                    [3, 0],
                    [4, 3],
                    [5, 4],
                    [6, 5],
                    [7, 6],
                    [8, 7],
                    [6, 8],
                ]
            )
            .transpose()
            .reshape(1, 2, -1)
            .repeat(len(X_train), axis=0)
        )
        # copy test set and randomly rotate globally
        X_val = X_train.copy()
        rng = np.random.default_rng(42)
        index = rng.integers(12, size=len(X_val))
        X_val = np.einsum("kij,kjl->kil", X_val, rotations[index])
        y_val = y_train.copy()

        coords = np.concatenate([X_train, X_val], axis=0)
        labels = np.concatenate([y_train, y_val], axis=0)
        adj = np.concatenate([edge_index, edge_index], axis=0)
        split = np.array([[0]] * len(X_train) + [[1]] * len(X_val))

        # create records
        def generator():
            for coord, label, edge_index in zip(coords, labels, adj):
                coord = coord.astype(np.float32)
                data = {
                    "molecule_label": [[label]],
                    "molecule_graph": [[edge_index]],
                    "atom_pos": [[[[coord]]]],
                }
                yield ak.Record(data)

        batches = batched(IteratorWithLength(generator(), len(coords)))
        return batches, Split(split, names=["train", "val"]), Assets({})
