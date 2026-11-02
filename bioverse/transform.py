import hashlib
import inspect
from typing import Iterator, Tuple

import awkward as ak
import numpy as np

from .data import Assets, Batch, Split
from .utilities import IteratorWithLength, note, parallelize, progressbar, rebatch


class Transform:
    """Modify batches, splits, and assets in a scikit-learn-style pipeline.

    Transforms follow a ``fit`` → ``transform`` lifecycle. Offline transforms
    are applied via :meth:`~bioverse.dataset.Dataset.apply` and materialized
    to disk; live transforms run at load time via
    :meth:`~bioverse.dataset.Dataset.live`.

    Override :meth:`transform_batch`, :meth:`transform_split`, and/or
    :meth:`transform_assets` to implement specific operations. Set
    :attr:`filter` to ``"scenes"`` to drop scenes marked by a
    ``scene_filter`` column.

    Examples
    --------
    Offline transform applied when building a dataset:

    .. code-block:: python

       from bioverse.transforms import TokenizeResidues, KnnGraph

       dataset.apply(TokenizeResidues(), KnnGraph(k=16))

    Live augmentation attached to a benchmark:

    .. code-block:: python

       from bioverse.transforms import Random2DRotate

       benchmark.live(Random2DRotate())
    """

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
        n_removed = 0
        n_total = 0
        masks: list[ak.Array] = []
        filtered_split = split

        def generate_filtered():
            nonlocal n_removed, n_total, filtered_split
            for batch in progressbar(
                batches,
                description="Filtering scenes",
                total=len(batches) if hasattr(batches, "__len__") else None,
            ):
                mask = batch.scene_filter
                masks.append(mask)
                n_total += len(mask)
                n_removed += int(ak.sum(~mask))
                batch.data.pop("scene_filter", None)
                yield batch.filter_scenes(mask)

            if split.data and masks and n_removed > 0:
                mask_np = ak.to_numpy(ak.concatenate(masks, axis=0))
                filtered_split = split.mask_filter(mask_np)
            _note_scene_filter(n_removed, n_total, split, masks)

        batches = rebatch(generate_filtered())
        return batches, filtered_split, assets

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


def _note_scene_filter(
    n_removed: int,
    n_total: int,
    split: Split,
    masks: list[ak.Array],
) -> None:
    n_kept = n_total - n_removed
    if n_removed == 0:
        note(f"Scene filter: kept all {n_total} scenes.")
        return

    msg = (
        f"Scene filter: removed {n_removed} of {n_total} scenes "
        f"({n_kept} kept)."
    )
    if split.data and masks:
        mask = ak.concatenate(masks, axis=0)
        mask_np = ak.to_numpy(mask)
        removed = split.mask_filter(~mask_np)
        split_key = split.default
        removed_labels = removed.data[split_key]
        unique, counts = np.unique(removed_labels, return_counts=True)
        partition_names = split.names[split_key]
        breakdown = ", ".join(
            f"{partition_names[i]}: {count}" for i, count in zip(unique, counts)
        )
        msg = f"{msg} Removed by partition: {breakdown}."
    note(msg)


class Compose(Transform):
    """Apply a sequence of transforms in order.

    Examples
    --------
    .. code-block:: python

       from bioverse.transform import Compose
       from bioverse.transforms import TokenizeResidues, KnnGraph

       pipeline = Compose(TokenizeResidues(), KnnGraph(k=16))
       batches, split, assets = pipeline(batches, split, assets)
    """

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

    def transform_assets(self, assets: Assets) -> Assets:
        for transform in self.transforms:
            assets = transform.transform_assets(assets)
        return assets

    def inverse_transform(self, y: ak.Array) -> ak.Array:
        for transform in self.transforms:
            y = transform.inverse_transform(y)
        return y


def flatten_transforms(*transforms: Transform) -> tuple[Transform, ...]:
    """Unwrap nested :class:`Compose` so apply/live hash the inner pipeline."""
    flat: list[Transform] = []
    for transform in transforms:
        if isinstance(transform, Compose):
            flat.extend(flatten_transforms(*transform.transforms))
        else:
            flat.append(transform)
    return tuple(flat)
