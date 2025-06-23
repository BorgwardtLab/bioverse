import bz2
import glob
import gzip
import itertools
import json
import lzma
import os
import pickle
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterator, Sequence
from itertools import islice
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Tuple, cast

import awkward as ak
import numpy as np
import requests
import yaml
from joblib import Parallel, delayed
from multiprocess import Pool
from rich.console import Console
from rich.progress import Progress
from rich.traceback import install as install_traceback

from ..data import Batch
from . import config
from .array import flatten
from .constants import ONE_TO_THREE, SHARD_SIZE

install_traceback(show_locals=False)

console = Console()


def save(obj: Any, path: Path | str):
    path = Path(path)
    os.makedirs(path.parents[0], exist_ok=True)
    if path.suffix == ".json.gz":
        with gzip.open(path, "w") as file:
            file.write(json.dumps(obj).encode("utf-8"))
    elif path.suffix == ".json":
        with open(path, "w") as file:
            json.dump(obj, file)
    elif path.suffix == ".npy":
        np.save(path, obj)
    elif path.suffix == ".ak":
        # ak.to_parquet(obj, path)
        with open(path, "wb") as file:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    elif path.suffix == ".pkl":
        with open(path, "wb") as file:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    elif path.suffix in [".yml", ".yaml"]:
        with open(path, "w") as file:
            yaml.dump(obj, file)
    else:
        raise ValueError(f"File extension {path.suffix} not known.")


def load(path: Path | str, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        if default is None:
            raise FileNotFoundError(f"File {path} does not exist.")
        else:
            return default
    if path.suffix == ".json.gz":
        with gzip.open(path, "r") as file:
            return json.loads(file.read().decode("utf-8"))
    elif path.suffix == ".json":
        with open(path, "r") as file:
            return json.load(file)
    elif path.suffix == ".jsonl":
        with open(path, "r") as file:
            return [json.loads(line) for line in file.readlines()]
    elif path.suffix == ".npy":
        return np.load(path)
    elif path.suffix == ".ak":
        # return ak.from_parquet(path)
        with open(path, "rb") as handle:
            return pickle.load(handle)
    elif path.suffix == ".pkl":
        with open(path, "rb") as handle:
            return pickle.load(handle)
    elif path.suffix in [".yml", ".yaml"]:
        with open(path, "r") as file:
            return yaml.safe_load(file)
    else:
        raise ValueError(f"File extension {path.suffix} not known.")


class IteratorWithLength(Iterator):
    def __init__(self, iterator: Iterator, length: int | None = None):
        if length is None:
            iterator, copy = itertools.tee(iterator)
            length = iterator_length(copy)
        self.iterator = iterator
        self.length = length

    def __next__(self):
        self.length -= 1
        if self.length < 0:
            raise StopIteration
        return next(self.iterator)

    def __len__(self) -> int:
        return self.length

    def copy(self):
        a, b = itertools.tee(self.iterator)
        self.iterator = a
        return IteratorWithLength(b, self.length)

    def map(self, fn):
        return IteratorWithLength(map(fn, self.iterator), self.length)


def batched(iterator: IteratorWithLength) -> IteratorWithLength:
    def generate_batches():
        while shard := list(islice(iterator, 0, SHARD_SIZE)):
            yield Batch(ak.Array(shard))

    return IteratorWithLength(
        generate_batches(), length=ceil(len(iterator) / SHARD_SIZE)
    )


def rebatch(iterator: Iterator[Batch]) -> IteratorWithLength:
    def generate_batches():
        shard = next(iterator)
        for item in iterator:
            diff = SHARD_SIZE - len(shard)
            shard += item[:diff]
            if len(shard) == SHARD_SIZE:
                yield shard
                shard = item[diff:]
        if len(shard) > 0:
            yield shard

    return IteratorWithLength(generate_batches())


def save_shards(iterator: Iterator[Batch], path: Path | str) -> None:
    path = Path(path)
    shard_num, scene_nums, toc = 0, [], []
    for shard in progressbar(iterator, description="Transforming"):
        shard_num += 1
        # shard_toc = {
        #     "shard": np.full(len(shard), shard_num),
        #     "frames": ak.num(shard, axis=1),
        #     "molecules": ak.num(shard, axis=2),
        # }
        # if "chains" in shard.fields:
        #     shard_toc["chains"] = ak.num(shard.chains, axis=3)
        #     if "residues" in shard.chains.fields:
        #         shard_toc["residues"] = ak.num(shard.chains.residues, axis=4)
        # if "adjacency" in shard.fields:
        #     shard_toc["edges"] = ak.num(shard.adjacency, axis=4)[:, :, :, 0]
        # shard.toc["shard"] = np.full(len(shard), shard_num)
        if "molecule_graph" in shard.data:
            graphs = shard.scenes.frames.molecules.molecule_graph
            shard.toc["graph"] = ak.num(graphs, axis=4)[:, :, :, 0]
        scene_nums.append(shard.toc.pop("scene"))
        toc.append(ak.Array(shard.toc))
        save(shard.data, path / f"{shard_num}.ak")
    if shard_num == 0:
        raise ValueError("No data to save!")
    save(shard_num, path / "num_shards.json")
    # toc = dict(zip(ak.fields(toc), ak.unzip(toc)))
    # toc = {k: ak.concatenate(v, axis=0) if len(v) > 1 else v[0] for k, v in toc.items()}
    # toc["scene"] = np.arange(len(next(iter(toc.values()))))
    toc = ak.concatenate(toc, axis=0) if shard_num > 1 else toc[0]
    toc["scene"] = np.arange(len(toc))
    save(toc, path / "toc.ak")
    save(ak.Array(scene_nums), path / "tos.ak")


def warn(msg: str) -> None:
    console.print(f"Warning: {msg}", style="yellow")


def info(msg: str) -> None:
    console.print(f"Info: {msg}", style="cyan")


def note(msg: str) -> None:
    console.print(f"Note: {msg}", style="#666666")


def progressbar(
    iterable: Iterable[Any], total: int | None = None, description: str = ">"
) -> Iterator[Any]:
    progress = Progress(transient=True)
    if total is None:
        if hasattr(iterable, "__len__"):
            total = len(iterable)  # type: ignore
    task = progress.add_task(description=description, total=total)
    t0 = time.time()
    with progress:
        for item in iterable:
            yield item
            progress.update(task, advance=1)
    time_taken = time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))
    note(f"{description} {time_taken}")


def parallelize(
    fn: Callable,
    data: Iterable,
    description: str = ">",
    progress: bool = True,
    total: int | None = None,
    max_workers: int | None = None,
) -> Iterator[Any]:
    workers = min(max_workers, config.workers) if max_workers else config.workers
    if workers == 1:
        job = (fn(item) for item in data)
    else:
        # job = Parallel(n_jobs=workers, return_as="generator")(
        #     delayed(fn)(item) for item in data
        # )
        def job():
            with Pool(processes=workers) as pool:
                for result in pool.imap(fn, data):
                    yield result

        job = job()
    if progress:
        progress_bar = Progress(transient=True)
        if total is None:
            if hasattr(data, "__len__"):
                total = len(data)  # type: ignore
        task = progress_bar.add_task(description=description, total=total)
        t0 = time.time()
        with progress_bar:
            for item in job:
                yield item
                progress_bar.update(task, advance=1)
        time_taken = time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))
        note(f"{description} {time_taken}")
    else:
        for item in job:
            yield item


def zip_file(
    src: Path | str, dest: Path | str | None = None, remove: bool = False
) -> Path:
    src = Path(src)
    if dest is None:
        dest = src.with_suffix(".gz")
    dest = Path(dest)
    with open(src, "rb") as f_in:
        with gzip.open(dest, "wb") as f_out:
            f_out.writelines(f_in)
    if remove:
        os.remove(src)
    return dest


def unzip_file(path: Path | str, remove: bool = True) -> Path:
    path = Path(path)
    if (path.parent / path.stem).exists():
        note(f"{path.stem} already unzipped.")
        return path.parent / path.stem
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f_in:
            with open(path.parent / path.stem, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    elif path.suffix == ".bz2":
        with bz2.open(path, "rb") as f_in:
            with open(path.parent / path.stem, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    elif path.suffix == ".zip":
        with zipfile.ZipFile(path, "r") as f_in:
            f_in.extractall(path.parent / path.stem)
    elif path.suffix == ".Z":
        subprocess.run(
            ["uncompress", "-c", path],
            check=True,
            stdout=open(path.parent / path.stem, "wb"),
        )
    elif path.suffix == ".xz":
        with lzma.open(path, "rb") as f_in:
            with open(path.parent / path.stem, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    if remove:
        os.remove(path)
    return path.parent / path.stem


def download(
    url: str,
    out_path: Path | str,
    chunk_size: int = 1024 * 1024,
    extension: str | None = None,
) -> None:
    url_path, out_path = Path(url.split("?")[0]), Path(out_path)
    if out_path.exists():
        note(f"{out_path.name} already downloaded.")
        return
    if not extension is None:
        out_path = out_path.with_suffix(extension)
    else:
        out_path = out_path.with_suffix("".join(url_path.suffixes))
    os.makedirs(out_path.parent, exist_ok=True)
    r = requests.get(url, stream=True, headers={"User-Agent": "XY"})
    r.raise_for_status()
    try:
        with open(out_path, "wb") as file:
            for data in progressbar(
                r.iter_content(chunk_size=chunk_size),
                description=f"Downloading {url_path.name}",
                total=int(r.headers.get("content-length", 0)) // chunk_size,
            ):
                file.write(data)
    except:
        os.remove(out_path)
        raise Exception(f"Could not download {url_path.name}.")
    if out_path.suffix in [".zip", ".gz", ".bz2", ".Z", ".xz"]:
        note(f"Unzipping {out_path}")
        out_path = unzip_file(out_path)
    if out_path.suffix == ".tar":
        note(f"Extracting {out_path}")
        extract(out_path)


def extract(
    tar_path: Path | str,
    out_path: Path | str | None = None,
    extract_members: bool = True,
    strip: int = 0,
    remove: bool = True,
) -> None:
    tar_path = Path(tar_path)
    if out_path is None:
        out_path = tar_path.parent / tar_path.stem
    out_path = Path(out_path)

    if out_path.exists():
        note(f"{out_path.name} already extracted.")
        return

    def get_members(file):
        for member in file.getmembers():
            parts = Path(member.path).parts
            member.path = Path(*parts[min(strip, len(parts) - 1) :])
            yield member

    out_path = Path(out_path)
    try:
        with tarfile.open(tar_path, "r") as file:
            len_members = len(file.getmembers())
            members = get_members(file)
            if extract_members:
                for member in progressbar(
                    members, description="Extracting", total=len_members
                ):
                    file.extract(member, out_path)
            else:
                file.extractall(
                    out_path,
                    members=progressbar(
                        file, description="Extracting", total=len_members
                    ),
                )
    except:
        os.remove(out_path)
        raise Exception(f"Could not extract {tar_path}.")
    if remove:
        os.remove(tar_path)


def glob_delete(pattern: str) -> None:
    for path in glob.iglob(str(pattern)):
        os.remove(path)


def iterator_length(iterator: Iterator) -> int:
    counter = itertools.count()
    deque(zip(iterator, counter), maxlen=0)
    return next(counter)


def alias(name: str) -> Callable[[Callable], Callable]:
    def decorator(cls: Callable) -> Callable:
        current_module = sys.modules[cls.__module__]
        setattr(current_module, name, cls)
        return cls

    return decorator


def interleave(*lists: Iterable) -> list:
    return list(itertools.chain.from_iterable(zip(*lists)))


def to_pdb(
    atom_pos, atom_label, residue_label=None, chain_label=None, pLDDT=None, **kwargs
) -> str:
    xyz = flatten(atom_pos, exclude=-1)
    if not residue_label is None:
        residue_numbers = [
            np.arange(1, len(chain) + 1)
            for chain in ak.broadcast_arrays(residue_label, atom_label)[0]
        ]
        residue_numbers = ak.ravel(ak.broadcast_arrays(residue_numbers, atom_label)[0])
        residue_label = ak.ravel(ak.broadcast_arrays(residue_label, atom_label)[0])
        # residue_label = [ONE_TO_THREE[res] if res in ONE_TO_THREE else res for res in residue_label]
        atom_numbers = np.concatenate(
            [np.arange(1, n + 1) for n in ak.run_lengths(residue_numbers)]
        )
    else:
        residue_label = ["XXX"] * len(xyz)
        residue_numbers = [1] * len(xyz)
        atom_numbers = [1] * len(xyz)
    if not chain_label is None:
        chain_label = ak.ravel(ak.broadcast_arrays(chain_label, atom_pos)[0])
    else:
        chain_label = ["A"] * len(xyz)
    if not pLDDT is None:
        pLDDT = ak.ravel(ak.broadcast_arrays(pLDDT, atom_pos)[0])
    else:
        pLDDT = [0.0] * len(xyz)
    atmn, occ = 0, 1.0
    lines = []
    atom_label = ak.ravel(atom_label)
    for atm, atmn, (x, y, z), res, resn, temp, chn in zip(
        atom_label,
        atom_numbers,
        xyz,
        residue_label,
        residue_numbers,
        pLDDT,
        chain_label,
    ):
        alt = ""
        icode = ""
        segid = ""
        elem = ""
        charge = ""
        lines.append(
            f"ATOM  {atmn:5d} {atm:<4s}{alt:1s}{res:>3s} {chn[-1]:1s}{resn:4d}{icode:1s}   {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{temp:6.2f}          {segid:<4s}{elem:>2s}{charge:2s}\n"
        )
    lines.append(f"TER   {atmn:5d}      {res:>3s} {chn[-1]:1s}{resn:4d}\n")
    return "".join(lines)


class LRUCache(OrderedDict):
    def __init__(self, size: int = config.cache_size):
        super().__init__()
        self.size = size
        self._check_size()

    def __setitem__(self, shard: int, value: Batch):
        super().__setitem__(shard, value)
        self._check_size()

    def _check_size(self):
        while len(self) > self.size:
            self.popitem(last=False)  # Remove oldest item
