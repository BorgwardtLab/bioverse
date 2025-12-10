import hashlib
import inspect
from typing import Iterator, Tuple

import awkward as ak
import numpy as np

from .data import Assets, Batch, Split
from .utilities import IteratorWithLength, note, parallelize, rebatch


class Transform:
    filter: bool | str = False

    def __call__(
        self, batches: Iterator[Batch], split: Split, assets: Assets
    ) -> Tuple[Iterator[Batch], Split, Assets]:
        self.fit(batches.copy(), split, assets)
        batches, split, assets = self.transform(batches, split, assets)
        if self.filter == "scenes":
            batches, split, assets = self.filter_scenes(batches, split, assets)
        return batches, split, assets

    def filter_scenes(
        self, batches: Iterator[Batch], split: Split, assets: Assets
    ) -> Tuple[Iterator[Batch], Split, Assets]:
        masks = [batch.scene_filter for batch in batches.copy()]

        def generator():
            for batch, mask in zip(batches, masks):
                batch.data.pop("scene_filter")
                yield batch[mask]

        batches = rebatch(generator())
        mask = ak.concatenate(masks, axis=0)
        if not len(split) == 0:
            removed = split[~mask]
            split = split[mask]
            unique, counts = np.unique(removed.ravel(), return_counts=True)
            names = [split.attrs["names"][i] for i in unique]
            out_str = ", ".join(
                f"{name}: {count}" for name, count in zip(names, counts)
            )
            note(f"Filtered {ak.sum(~mask)} of {len(mask)} scenes ({out_str}).")
        else:
            note(f"Filtered {ak.sum(~mask)} of {len(mask)} scenes.")
        return batches, split, assets

    def fit(self, batches: Iterator[Batch], split: Split, assets: Assets) -> None:
        pass

    def transform(
        self, batches: Iterator[Batch], split: Split, assets: Assets
    ) -> Tuple[Iterator[Batch], Split, Assets]:
        assets = self.transform_assets(assets)
        split = self.transform_split(split)
        batches = IteratorWithLength(
            parallelize(
                self.transform_batch,
                batches,
                description=self.__class__.__name__,
                progress=False,
            ),
            length=len(batches),
        )
        return batches, split, assets

    def transform_batch(self, batch: Batch) -> Batch:
        return batch

    def transform_split(self, split: Split) -> Split:
        return split

    def transform_assets(self, assets: Assets) -> Assets:
        return assets

    def inverse_transform(self, y: ak.Array) -> ak.Array:
        return y

    def hash(self) -> str:
        h = hashlib.sha256()
        h.update(str(self.__class__).encode())
        h.update(b"|")

        # Get argument names and values using inspect
        arg_spec = inspect.getfullargspec(self.__init__)
        args = arg_spec.args[1:]  # Skip "self"

        # Create a sorted list of key-value pairs for arguments
        arg_vals = [(arg, getattr(self, arg)) for arg in args]
        sorted_args = sorted(arg_vals)

        # Update hash with string representation of arguments
        h.update(repr(sorted_args).encode())
        return h.hexdigest()


class Compose(Transform):

    def __init__(self, *transforms: Transform):
        self.transforms = transforms

    def hash(self) -> str:
        h = hashlib.sha256()
        for transform in self.transforms:
            h.update(transform.hash().encode())
        return h.hexdigest()

    def __call__(
        self, batches: Iterator[Batch], split: Split, assets: Assets
    ) -> Tuple[Iterator[Batch], Split, Assets]:
        for transform in self.transforms:
            batches, split, assets = transform(batches, split, assets)
        return batches, split, assets

    def transform_batch(self, batch: Batch) -> Batch:
        for transform in self.transforms:
            batch = transform.transform_batch(batch)
        return batch

    def inverse_transform(self, y: ak.Array) -> ak.Array:
        for transform in self.transforms:
            y = transform.inverse_transform(y)
        return y
