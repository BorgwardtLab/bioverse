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


class Benchmark(ABC):

    dataset: Dataset
    sampler: Sampler
    task: Task
    metric: Metric

    def __init__(
        self,
        root: Path | str = config.benchmarks_path,
        version: int = 0,
        partition: str = "main",
        n_jobs: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        root : Path or str, optional
            Root directory for benchmark data storage, defaults to config.benchmarks_path
        version : int, optional
            Version number of the benchmark, defaults to 0
        partition : str, optional
            Partition name for the benchmark data, defaults to "main"
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
        self.root = Path(root) / f"{self.name}v{version}" / partition
        os.makedirs(self.root, exist_ok=True)
        self.partition = partition
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
        for split in self.dataset.split.attrs["names"]:
            setattr(self, f"{split}_loader", partial(self.wrap_loader, split=split))
        return self

    def wrap_loader(self, *args, **kwargs):
        if not "random_seed" in kwargs or kwargs["random_seed"] is None:
            kwargs["random_seed"] = config.seed

        class Loader:
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
        split: str,
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

        if not split in self.dataset.split.attrs["names"]:
            return None

        if scratch:
            self.dataset.move_to_scratch()

        # sample the index
        index = self.sampler.sample(
            dataset=self.dataset,
            split=split,
            batch_size=batch_size,
            batch_on=batch_on,
            shuffle=shuffle,
            drop_last=drop_last,
            random_seed=random_seed,
            world_size=world_size,
            rank=rank,
        )

        task, vbatch, assets = self.task, self.dataset.virtual(), self.dataset.assets

        def worker(batch_index):
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
        self.metric.update(y_true, y_pred)

    def result(self, *args, **kwargs) -> Result:
        return self.leaderboard + self.metric.result(*args, **kwargs)
