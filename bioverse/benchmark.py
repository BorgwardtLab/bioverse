import inspect
import os
from abc import ABC
from functools import partial
from pathlib import Path
from typing import Iterable, Tuple

import awkward as ak
from typing_extensions import Self

from .collater import Collater
from .data import Batch
from .dataset import Dataset
from .metric import Metric, Result
from .sampler import Sampler
from .task import Task
from .transform import Transform
from .utilities import config, load, parallelize
from .virtual import VirtualBatch


class Benchmark(ABC):
    """Orchestrate dataset loading, task extraction, and metric evaluation.

    A benchmark binds four components declared as class attributes:

    * :class:`~bioverse.dataset.Dataset`
    * :class:`~bioverse.sampler.Sampler`
    * :class:`~bioverse.task.Task`
    * :class:`~bioverse.metric.Metric`

    :meth:`loader` drives the full path from sampled indices to ``(X, y)``
    pairs (and optionally collated batches). The
    :class:`~bioverse.trainer.Trainer` iterates loaders and calls
    :meth:`update` / :meth:`result` during evaluation.

    Examples
    --------
    Using a benchmark config (``B_*.yaml``):

    .. code-block:: python

       from bioverse.factory import BenchmarkFactory

       benchmark = BenchmarkFactory("B_AFCATH")
       for (X, y), collated in benchmark.loader("train", batch_size=32):
           ...

    Attaching live transforms:

    .. code-block:: python

       from bioverse.transforms import Random2DRotate

       benchmark.live(Random2DRotate())
    """

    dataset: Dataset
    sampler: Sampler
    task: Task
    metric: Metric

    def __init__(
        self,
        root: Path | str = config.benchmarks_path,
        version: int = 0,
        split: str = "default",
        n_jobs: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        root : Path or str, optional
            Root directory for benchmark data storage, defaults to config.benchmarks_path
        version : int, optional
            Version number of the benchmark, defaults to 0
        split : str or None, optional
            Split name for the benchmark data, defaults to None
        n_jobs : int or None, optional
            Number of parallel jobs to run. If None, uses all available cores

        Notes
        -----
        The benchmark class requires the following class attributes to be defined:
        - dataset: Dataset class, instance, or (class, kwargs) tuple
        - sampler: Sampler class, instance, or (class, kwargs) tuple
        - task: Task class, instance, or (class, kwargs) tuple
        - metric: Metric class, instance, or (class, kwargs) tuple
        """
        self.root = Path(root) / f"{self.name}v{version}" / split
        os.makedirs(self.root, exist_ok=True)
        self.n_jobs = n_jobs

        # initialize components from config, class, or instance
        for attr in ["dataset", "sampler", "task", "metric"]:
            kwargs = getattr(self, attr)
            if isinstance(kwargs, tuple):
                cls, kwargs = kwargs
                setattr(self, attr, cls(**kwargs))
            elif inspect.isclass(kwargs):
                setattr(self, attr, kwargs())
            else:
                setattr(self, attr, kwargs)

        self.split = self.dataset.split.default if split == "default" else split

    def _partitions(self):
        if self.split in self.dataset.split.names:
            return self.dataset.split.names[self.split]
        if self.dataset.split.names:
            return next(iter(self.dataset.split.names.values()))
        return ["train"]

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def leaderboard(self) -> Result:
        # temporary fix for leaderboard database
        file_path = os.path.realpath(os.path.dirname(__file__))
        return Result(load(f"{file_path}/leaderboards/{self.name}.yml", []))

    def apply(self, *transforms: Transform) -> Self:
        self.dataset.apply(*transforms)
        # initialize loaders (transforms can change splits, so we do it here)
        for partition in self._partitions():
            setattr(
                self,
                f"{partition}_loader",
                partial(self.wrap_loader, partition=partition),
            )
        return self

    def live(self, *transforms: Transform) -> Self:
        self.dataset.live(*transforms)
        return self

    def wrap_loader(self, *args, **kwargs):
        if not "random_seed" in kwargs or kwargs["random_seed"] is None:
            kwargs["random_seed"] = config.seed

        class Loader:
            """Stateful wrapper that increments the random seed on each iteration."""

            def __init__(self, loader):
                self.loader = loader
                self.args = args
                self.kwargs = kwargs

            def __iter__(self):
                self.kwargs["random_seed"] += 1
                return self.loader(*self.args, **self.kwargs)

        return Loader(self.loader)

    def loader(
        self,
        partition: str,
        collater: Collater = None,
        batch_size: int = 1,
        batch_on: str = "molecules",
        shuffle: bool = False,
        drop_last: bool = False,
        random_seed: int = config.seed,
        world_size: int = 1,
        rank: int = 0,
        progress: bool = True,
        attr: list[str] = [],
        scratch: bool = False,
    ) -> Iterable[Tuple[Tuple[ak.Array, ...], Batch | None]]:

        if not partition in self._partitions():
            return None

        if scratch:
            self.dataset.move_to_scratch()

        # sample the index
        index = self.sampler.sample(
            dataset=self.dataset,
            split=self.split,
            partition=partition,
            batch_size=batch_size,
            batch_on=batch_on,
            shuffle=shuffle,
            drop_last=drop_last,
            random_seed=random_seed,
            world_size=world_size,
            rank=rank,
        )

        task, assets = self.task, self.dataset.assets
        path = self.dataset.path
        live_transform = self.dataset.live_transform

        # todo: maybe apply live transforms to assets and splits

        def worker(batch_index):
            vbatch = VirtualBatch(path, assets, live_transform)
            Xy = task(vbatch, assets, batch_index)
            data = collater(*Xy, attr=attr, assets=assets) if collater else None
            return Xy, data

        return parallelize(
            worker,
            index,
            description="Loader",
            progress=progress and rank == 0,
            total=len(index),
            max_workers=10,
        )

    def update(self, y_true: ak.Array, y_pred: ak.Array) -> None:
        y_true = self.dataset.transform.inverse_transform(y_true)
        y_pred = self.dataset.transform.inverse_transform(y_pred)
        y_true = self.dataset.live_transform.inverse_transform(y_true)
        y_pred = self.dataset.live_transform.inverse_transform(y_pred)
        self.metric.update(y_true, y_pred)

    def result(self, *args, **kwargs) -> Result:
        return self.leaderboard + self.metric.result(*args, **kwargs)
