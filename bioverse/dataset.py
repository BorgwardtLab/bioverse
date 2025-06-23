from __future__ import annotations

import itertools
import os
import shutil
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Iterator, Tuple

import awkward as ak

from .data import Assets, Batch, Split
from .transform import Compose, Transform
from .transforms import Identity
from .utilities import (
    SHARD_SIZE,
    IteratorWithLength,
    config,
    info,
    load,
    note,
    rebatch,
    save,
    save_shards,
)
from .virtual import VirtualBatch


class Dataset(ABC):

    def __init__(
        self,
        root: Path | str = config.dataset_path,
        version: int | None = None,
        online: bool = True,
    ) -> None:
        self.root = Path(root)
        self.croot = Path(root) / self.__class__.__name__
        self.online = online
        self.transform = Compose(Identity())
        if version is None:
            if online and not self.latest_online_version is None:
                version = self.latest_online_version
                if not os.path.exists(self.croot / f"v{version}"):
                    info(f"There is a newer version {version} available.")
            else:
                local_version = self.latest_local_version
                if not local_version is None:
                    info(f"Loading latest local version {local_version}.")
                    version = local_version
        version = version or self.bump_version
        if not os.path.exists(self.croot / f"v{version}") and not (
            online and not self.download(version) is None
        ):
            info(f"Could not find dataset. Running a release.")
            self.run_release(version)
            info(f"Released version {version}.")
        self.version = version
        self.path = self.croot / f"v{version}" / self.transform.hash()
        self.clear_property_caches()
        if not os.path.exists(self.path / "transform.pkl"):
            save(self.transform, self.path / "transform.pkl")

    @property
    def latest_local_version(self) -> int | None:
        if not os.path.exists(self.croot):
            return None
        versions = [
            int(entry.name[1:]) for entry in os.scandir(self.croot) if entry.is_dir()
        ]
        return max(versions) if len(versions) > 0 else None

    @property
    def bump_version(self) -> int:
        return (self.latest_local_version or 0) + 1

    @property
    def latest_online_version(self) -> int | None:
        return None

    def download(self, version: int) -> int | None:
        # if dataset is not hosted or version does not exist online: return None
        # download version to self.croot
        return None

    def save(
        self,
        data: Iterator[Batch],
        split: Split,
        assets: Assets,
        version: int,
    ) -> int:
        if os.path.exists(self.croot / f"v{version}"):
            raise Exception(f"Version {version} already exists!")
        self.version = version
        self.path = self.croot / f"v{version}" / self.transform.hash()
        self.clear_property_caches()
        save_shards(data, self.path)
        save(split, self.path / "split.ak")
        save(split.attrs["names"], self.path / "split.names.json")
        save(assets, self.path / "assets.json")
        os.makedirs(self.path, exist_ok=True)
        return version

    @cached_property
    def assets(self) -> Assets:
        return load(self.path / "assets.json")

    @cached_property
    def toc(self) -> ak.Array:
        return load(self.path / "toc.ak")

    @cached_property
    def tos(self) -> ak.Array:
        return load(self.path / "tos.ak")

    @cached_property
    def split(self) -> Split:
        index = load(self.path / "split.ak")
        names = load(self.path / "split.names.json")
        return Split(index, names)

    @cached_property
    def num_shards(self) -> int:
        return load(self.path / "num_shards.json")

    def clear_property_caches(self) -> None:
        for attr in ["assets", "toc", "split", "num_shards", "data"]:
            self.__dict__.pop(attr, None)

    def __len__(self) -> int:
        last_shard = load(self.path / f"{self.num_shards}.shard")
        return SHARD_SIZE * (self.num_shards - 1) + len(last_shard)

    @property
    def shards(self) -> Iterator[Batch]:
        return IteratorWithLength(
            (
                Batch(load(self.path / f"{shard+1}.ak"))
                for shard in range(self.num_shards)
            ),
            length=self.num_shards,
        )

    @cached_property
    def data(self):
        return self.virtual()

    def virtual(self):
        return VirtualBatch(self.path, self.assets)

    def apply(self, *transforms: Transform) -> None:
        self.transform = Compose(*transforms)
        new_path = self.croot / f"v{self.version}" / self.transform.hash()
        if not os.path.exists(new_path):
            shards, split, assets = self.transform(self.shards, self.split, self.assets)
            save_shards(shards, new_path)
            save(split, new_path / "split.ak")
            save(split.attrs["names"], new_path / "split.names.json")
            save(assets, new_path / "assets.json")
            save(self.transform, new_path / "transform.pkl")
        else:
            note("Data were already transformed. Loading from disk.")
            note(str(new_path))
            self.transform = load(new_path / "transform.pkl")  # for fitted values
        self.path = new_path
        self.clear_property_caches()

    @abstractmethod
    def release(self) -> Tuple[Iterator[Batch], Split, Assets]:
        raise NotImplementedError

    def package(self) -> None:
        pass

    def run_release(self, version: int) -> None:
        batches, split, assets = self.release()
        self.save(batches, split, assets, version)

    def citation(self, style: str = "apa") -> str:
        raise NotImplementedError

    def license(self) -> str:
        raise NotImplementedError

    def statistics(self) -> str:
        raise NotImplementedError

    def __add__(self, other: Dataset | ComposedDataset) -> ComposedDataset:
        return ComposedDataset(self, other)

    def move_to_scratch(self) -> None:
        if config.scratch_path == Path():
            raise ValueError("Please set config.scratch_path first.")
        scratch_path = (
            config.scratch_path
            / self.__class__.__name__
            / f"v{self.version}"
            / self.transform.hash()
        )
        if not scratch_path.exists():
            note(f"Copying to scratch.")
            scratch_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.path, scratch_path)
        self.path = scratch_path
        self.clear_property_caches()


class ComposedDataset:

    def __init__(self, *datasets: Dataset | ComposedDataset) -> None:
        self.shards = iter([])
        self.split = Split()
        self.assets = Assets()
        if len(datasets) > 0:
            for dataset in datasets:
                self.__add__(dataset)

    def __add__(self, other: Dataset | ComposedDataset) -> ComposedDataset:
        self.shards = rebatch(itertools.chain(self.shards, other.shards))
        other_names = [
            f"{x}_{other.__class__.__name__}" for x in other.split.attrs["names"]
        ]
        self.split = ak.concatenate(
            [self.split, other.split + len(self.split.attrs["names"])],  # type: ignore
            attrs={"names": self.split.attrs["names"] + other_names},
        )
        self.assets[other.__class__.__name__] = other.assets
        return self
