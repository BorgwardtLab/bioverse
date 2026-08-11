from __future__ import annotations

import itertools
import os
import shutil
import tarfile
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
    zip_file,
)
from .utilities.precomputed import Precomputed, fetch_precomputed
from .virtual import VirtualBatch


class Dataset(ABC):
    """Versioned, on-disk collection of sharded biomolecular data.

    Datasets persist batches under ``config.dataset_path/<Name>/v<version>/``.
    Offline :class:`~bioverse.transform.Transform` pipelines are materialized
    to a content-addressed subdirectory; live transforms run at load time.

    Subclasses implement :meth:`release` to produce batches from an adapter or
    upstream source. Dataset configs (``D_*.yaml``) reference adapters and
    transform lists.

    Examples
    --------
    .. code-block:: python

       from bioverse.factory import DatasetFactory

       dataset = DatasetFactory("D_AFCATH")
       dataset.apply(TokenizeResidues())
       print(len(dataset), dataset.split.names)
    """

    def __init__(
        self,
        root: Path | str = config.dataset_path,
        version: int | None = None,
        online: bool = True,
        precomputed: dict | Precomputed | None = None,
    ) -> None:
        self.root = Path(root)
        self.croot = Path(root) / self.name
        self.online = online
        self.precomputed = (
            precomputed
            if isinstance(precomputed, Precomputed) or precomputed is None
            else Precomputed(precomputed)
        )
        self.transform = Compose(Identity())
        self.live_transform = Compose(Identity())
        if version is None:
            if online and self.latest_online_version is not None:
                version = self.latest_online_version
                if not self._is_materialized(version):
                    info(f"There is a newer version {version} available online.")
            elif self.latest_local_version is not None:
                info(f"Loading latest local version {self.latest_local_version}.")
                version = self.latest_local_version
        version = version or self.bump_version
        self.version = version
        if not self._is_materialized(version):
            if online and self.download(version) is not None:
                info(f"Downloaded precomputed version {version}.")
            else:
                info("Could not find dataset. Running a release.")
                self.run_release(version)
                info(f"Released version {version}.")
        self.path = self.croot / f"v{version}" / self.transform.hash()
        self.clear_property_caches()
        if not os.path.exists(self.path / "transform.pkl"):
            save(self.transform, self.path / "transform.pkl")

    @property
    def name(self) -> str:
        return self.__class__.__name__

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
        if self.precomputed is None:
            return None
        return self.precomputed.latest_version

    def _materialized_path(self, version: int, transform_hash: str | None = None) -> Path:
        transform_hash = transform_hash or self.transform.hash()
        return self.croot / f"v{version}" / transform_hash

    def _is_materialized(
        self, version: int, transform_hash: str | None = None
    ) -> bool:
        return (self._materialized_path(version, transform_hash) / "num_shards.json").is_file()

    def download(self, version: int) -> int | None:
        if self._fetch_precomputed(version):
            return version
        return None

    def _fetch_precomputed(
        self, version: int, transform_hash: str | None = None
    ) -> bool:
        transform_hash = transform_hash or self.transform.hash()
        dest = self._materialized_path(version, transform_hash)
        return fetch_precomputed(self.precomputed, version, dest, transform_hash)

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
        split.save(self.path)
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
        return Split.load(self.path)

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
        return VirtualBatch(self.path, self.assets, self.live_transform)

    def apply(self, *transforms: Transform) -> None:
        self.transform = Compose(*transforms)
        new_path = self.croot / f"v{self.version}" / self.transform.hash()
        transform_path = new_path / "transform.pkl"
        if not self._is_materialized(self.version, self.transform.hash()):
            if self.online and self._fetch_precomputed(
                self.version, self.transform.hash()
            ):
                note("Downloaded precomputed transform.")
            else:
                if new_path.exists():
                    shutil.rmtree(new_path)
                shards, split, assets = self.transform(
                    self.shards, self.split, self.assets
                )
                save_shards(shards, new_path)
                split.save(new_path)
                save(assets, new_path / "assets.json")
                save(self.transform, transform_path)
        elif transform_path.is_file():
            note("Data were already transformed. Loading from disk.")
            note(str(new_path))
            self.transform = load(transform_path)  # for fitted values
        else:
            save(self.transform, transform_path)
        self.path = new_path
        self.clear_property_caches()

    def live(self, *transforms: Transform) -> None:
        self.live_transform = Compose(*transforms)

    @abstractmethod
    def release(self) -> Tuple[Iterator[Batch], Split, Assets]:
        raise NotImplementedError

    def package(self, dest: Path | str | None = None) -> Path:
        """Archive the current dataset version for upload to a repository."""
        if dest is None:
            dest = (
                self.croot
                / f"{self.name}_v{self.version}_{self.transform.hash()[:8]}.tar.gz"
            )
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        archive = dest.with_suffix(".tar")
        with tarfile.open(archive, "w") as tar:
            tar.add(self.path, arcname=self.transform.hash())
        zip_file(archive, dest, remove=True)
        note(f"Packaged dataset to {dest}")
        return dest

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
            config.scratch_path / self.name / f"v{self.version}" / self.transform.hash()
        )
        if not scratch_path.exists():
            note(f"Copying to scratch.")
            scratch_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.path, scratch_path)
        self.path = scratch_path
        self.clear_property_caches()


class ComposedDataset:
    """Concatenate multiple datasets into a single shard stream.

    Used internally when combining datasets with ``dataset_a + dataset_b``.
    Splits are not yet merged across constituents (see source TODO).
    """

    def __init__(self, *datasets: Dataset | ComposedDataset) -> None:
        self.shards = iter([])
        self.split = Split()
        self.assets = Assets()
        if len(datasets) > 0:
            for dataset in datasets:
                self.__add__(dataset)

    def __add__(self, other: Dataset | ComposedDataset) -> ComposedDataset:
        self.shards = rebatch(itertools.chain(self.shards, other.shards))
        # todo: what about splits?
        self.assets[other.name] = other.assets
        return self
