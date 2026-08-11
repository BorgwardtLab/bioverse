from abc import ABC, abstractmethod
from typing import Tuple

import awkward as ak

from .data import Assets, Batch
from .virtual import VirtualBatch


class Task(ABC):
    """Extract model inputs and targets from indexed dataset rows.

    A task defines the prediction problem for a benchmark. Given a batch index
    produced by a :class:`~bioverse.sampler.Sampler`, it loads the relevant
    shard through a :class:`~bioverse.virtual.VirtualBatch`, selects features
    ``X`` and supervision ``y``, and returns both as Awkward Arrays.

    Subclasses implement :meth:`__call__`. The trainer passes the output to a
    :class:`~bioverse.collater.Collater` and the metric compares ``y`` with
    model predictions.

    Examples
    --------
    .. code-block:: python

       from bioverse.tasks import PropertyPredictionTask

       task = PropertyPredictionTask(property="affinity", level="molecule")
       X, y = task(vbatch, assets, index)
    """

    @abstractmethod
    def __call__(
        self, vbatch: VirtualBatch, assets: Assets, index: Tuple[ak.Array, ...]
    ) -> Tuple[Batch, ak.Array]:
        """Return features and targets for one batch index.

        Parameters
        ----------
        vbatch
            Lazy, cache-backed view over on-disk dataset shards.
        assets
            Shared lookup tables for the dataset.
        index
            Awkward arrays indexing scene, frame, molecule, etc.

        Returns
        -------
        X
            Input features as a :class:`~bioverse.data.Batch`.
        y
            Supervision targets as an Awkward Array.
        """
        raise NotImplementedError
