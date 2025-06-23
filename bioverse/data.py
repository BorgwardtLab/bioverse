from __future__ import annotations

from typing import Any, List, Tuple, cast

import awkward as ak
import numpy as np


class BatchProxy:
    def __init__(self, batch: Batch, nesting: List[int] = []):
        self.batch = batch
        self.nesting = nesting

    def __getattr__(self, name: str) -> BatchProxy | ak.Array:
        if name == "vertices":
            name = "residues" if self.batch.resolution == "residue" else "atoms"
        if name.startswith("vertex"):
            name = name.replace(
                "vertex", "residue" if self.batch.resolution == "residue" else "atom"
            )

        # if property name is another level in the hierarchy, increase nesting
        if name.rstrip("s") in self.batch.prefixes:
            return BatchProxy(
                self.batch,
                self.nesting + [self.batch.prefixes.index(name.rstrip("s")) + 1],
            )

        # if property name is a data column, flatten and return data according to nesting
        elif "_" in name and name.split("_")[0] in self.batch.prefixes:
            x = self.batch.data[name]
            for axis in range(1, self.batch.prefixes.index(name.split("_")[0]) + 1)[
                ::-1
            ]:
                if not axis in self.nesting:
                    x = x.flatten(axis=axis)  # type: ignore[attr-defined]
            x = cast(ak.Array, x)
            return x
        else:
            raise AttributeError(f"Attribute '{name}' not found in data.")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("vertex"):
            name = name.replace(
                "vertex", "residue" if self.batch.resolution == "residue" else "atom"
            )
        # if property name is a data column, unflatten and set data according to nesting
        if "_" in name and name.split("_")[0] in self.batch.prefixes:
            if not isinstance(value, ak.Array):
                value = ak.Array(value)
            prefix = name.split("_")[0]
            if (
                next(
                    (col for col in self.batch.data.keys() if col.startswith(prefix)),
                    None,
                )
                is None
                and prefix != "scene"
            ):  # no such key in the data, infer toc from value
                previous_prefix = self.batch.prefixes[
                    self.batch.prefixes.index(prefix) - 1
                ]
                num = self.batch.toc[previous_prefix]
                toc_vals = ak.num(value, axis=-1)
                if num.ndim > 0:
                    toc_vals = ak.unflatten(toc_vals, 1, axis=-1)
                    for i in range(num.ndim)[1:][::-1]:
                        toc_vals = ak.unflatten(toc_vals, ak.num(num, axis=i).ravel())
                self.batch.toc[prefix] = toc_vals
            for axis in range(1, self.batch.prefixes.index(prefix) + 1)[::-1]:
                if not axis in self.nesting:
                    # print(name, axis, self.batch.prefixes[axis], self.nesting)
                    # print(value, len(value))
                    # print(
                    #     self.batch.toc[self.batch.prefixes[axis]],
                    #     ak.sum(self.batch.toc[self.batch.prefixes[axis]]),
                    #     len(self.batch.toc[self.batch.prefixes[axis]].ravel()),
                    # )
                    # print(sum([n < axis for n in self.nesting]))
                    # print(self.batch.toc[self.batch.prefixes[axis]].to_list())
                    value = value.unflatten(  # type: ignore[attr-defined]
                        counts=self.batch.toc[self.batch.prefixes[axis]].ravel(),
                        axis=sum([n < axis for n in self.nesting]),
                    )
            self.batch.data[name] = value
        else:
            return super().__setattr__(name, value)


class Batch:

    def __init__(
        self,
        data: dict[str, ak.Array],
        prefixes: List[str] = [
            "scene",
            "frame",
            "molecule",
            "chain",
            "residue",
            "atom",
        ],
        resolution: str = "atom",
    ):
        if isinstance(data, ak.Array) or isinstance(data, ak.Record):
            data = {field: data[field] for field in data.fields}
        self.data = data
        self.resolution = resolution
        self.prefixes = prefixes
        self.toc = {}
        self.length = len(data[list(data.keys())[0]]) if len(data) > 0 else 0  # type: ignore[attr-defined]

        # determine the number of elements in each axis
        if data:
            for prefix in prefixes:
                ref_col = next(
                    (col for col in data.keys() if col.startswith(prefix)), None
                )
                if not ref_col is None:
                    counts = ak.num(data[ref_col], axis=prefixes.index(prefix))
                    # if not counts.ndim == 0:
                    #     counts = ak.ravel(counts)
                    self.toc[prefix] = counts
                elif prefix == "scene":
                    self.toc[prefix] = ak.num(data[list(data.keys())[0]], axis=0)
                else:
                    # assume the prefix is a single element
                    previous_prefix = prefixes[prefixes.index(prefix) - 1]
                    num = self.toc[previous_prefix]
                    ones = np.ones(num if num.ndim == 0 else ak.sum(num)).astype(int)
                    if num.ndim > 0:
                        # ones = ones.reshape(-1, 1)
                        ones = ak.unflatten(ones, num.ravel(), axis=0)
                        for i in range(num.ndim)[1:][::-1]:
                            ones = ak.unflatten(ones, ak.num(num, axis=i).ravel())
                    self.toc[prefix] = ones

    def __len__(self) -> int:
        return self.length

    def __contains__(self, item: str) -> bool:
        if item.startswith("vertex"):
            item = item.replace(
                "vertex", "residue" if self.resolution == "residue" else "atom"
            )
        return item in self.data

    def __getattr__(self, name: str) -> BatchProxy | ak.Array:
        if name in ["data", "prefixes", "levels", "toc", "length", "resolution"]:
            return super().__getattr__(name)
        else:
            return BatchProxy(self).__getattr__(name)

    def __setattr__(self, name: str, value: ak.Array) -> None:
        if "_" in name and name.split("_")[0] in self.prefixes + ["vertex"]:
            return BatchProxy(self).__setattr__(name, value)
        else:
            return super().__setattr__(name, value)

    def __getitem__(
        self, index: slice | int | ak.Array | Tuple[slice | int | ak.Array, ...]
    ) -> Batch:

        # the index is a tuple of indices for each axis
        if not isinstance(index, tuple):
            index = (index,)

        def deep_index(v, idx):
            for i in idx:
                v = v[i]
            return v

        # arrays are immutable, so we return a new one
        levels = [
            self.prefixes.index(field.split("_")[0]) + 1 for field in self.data.keys()
        ]
        data = {
            # limit the index to the depth of the data for each field
            field: deep_index(self.data[field], index[:level])  # type: ignore[attr-defined]
            for field, level in zip(self.data.keys(), levels)
        }
        return Batch(data)

    def __add__(self, other: Batch) -> Batch:
        self_data, other_data, both_data = self.data, other.data, {}
        for field in sorted(set(self_data.keys()) | set(other_data.keys())):
            prefix = field.split("_")[0]
            if not field in self_data:
                if prefix == "scene":
                    self_data[field] = ak.nan_to_none(
                        np.ones(self.toc[prefix]) * np.nan
                    )
                else:
                    self_data[field] = ak.nan_to_none(
                        ak.ones_like(self.toc[prefix]).unflatten(1, -1) * np.nan
                    )
            if not field in other_data:
                if prefix == "scene":
                    other_data[field] = ak.nan_to_none(
                        np.ones(other.toc[prefix]) * np.nan
                    )
                else:
                    other_data[field] = ak.nan_to_none(
                        ak.ones_like(other.toc[prefix]).unflatten(1, -1) * np.nan
                    )
            both_data[field] = ak.concatenate([self_data[field], other_data[field]])
        return Batch(both_data)

    def __radd__(self, other: Batch | Any) -> Batch:
        # if other is not a Batch, return a copy of self
        if not isinstance(other, Batch):
            data = {field: self.data[field] for field in self.data.keys()}
            return Batch(data)
        else:
            return other + self


class Split(ak.Array):

    def __init__(
        self,
        index: ak.Array | np.ndarray | list = [],
        names: List[str] = [],
        *args,
        **kwargs,
    ):
        if isinstance(index, ak.Array) and "names" in index.attrs:
            names = index.attrs["names"]
        super().__init__(index, attrs={"names": names}, *args, **kwargs)


class Assets(dict):
    pass
