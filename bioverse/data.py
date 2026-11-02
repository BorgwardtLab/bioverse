from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, List, Tuple, cast

import awkward as ak
import numpy as np


_TOC_REFERENCE_COLUMNS = {
    "molecule": ("molecule_id",),
    "chain": ("chain_label",),
    "residue": ("residue_number", "residue_label"),
    "atom": ("atom_label", "atom_pos"),
}


def _reference_column(data: dict[str, ak.Array], prefix: str) -> str | None:
    """Pick a stable, non-optional column for hierarchy toc counts."""
    for name in _TOC_REFERENCE_COLUMNS.get(prefix, ()):
        if name in data:
            return name
    for col, arr in data.items():
        if not (col.startswith(f"{prefix}_") or col == prefix):
            continue
        layout = arr.layout
        if getattr(layout, "is_option", False):
            continue
        if type(layout).__name__ == "UnionArray":
            continue
        return col
    return next((col for col in data if col.startswith(prefix)), None)


def _as_row_choice(indexer) -> np.ndarray:
    """Collapse a per-row frame/molecule indexer to a 1-d integer array."""
    if isinstance(indexer, (int, np.integer)):
        return np.asarray([int(indexer)], dtype=int)
    choice = indexer
    if isinstance(choice, ak.Array):
        while choice.ndim > 1:
            try:
                if ak.all(ak.num(choice, axis=1) == 1):
                    choice = choice[:, 0]
                    continue
            except Exception:
                pass
            choice = ak.flatten(choice, axis=1)
            break
        return np.asarray(ak.to_numpy(ak.ravel(choice)), dtype=int)
    return np.asarray(choice, dtype=int).reshape(-1)


def _take_next_axis(v, indexer):
    """Select along axis 1 of a batched array using a per-row indexer.

    Nested ``[[0], [0], ...]`` indexers must not re-slice axis 0. Doing so
    maps every row onto scene 0, which crashes mixed-scene mutation batches
    when a later row's mutation index exceeds scene 0's table length.
    The selected axis is kept as length 1 so Batch TOC can still count
    scene/frame/molecule levels after indexing.
    """
    if getattr(v, "ndim", 0) <= 1:
        return v[indexer]
    if isinstance(indexer, slice):
        return v[(slice(None), indexer)]
    if isinstance(indexer, (int, np.integer)):
        start = int(indexer)
        return v[:, start : start + 1]
    choice = _as_row_choice(indexer)
    n = len(v)
    if len(choice) == 1 and n != 1:
        choice = np.full(n, int(choice[0]), dtype=int)
    if len(choice) != n:
        return v[indexer]
    try:
        if np.all(choice == choice[0]):
            start = int(choice[0])
            return v[:, start : start + 1]
    except Exception:
        pass
    rows = []
    for i in range(n):
        row = v[i]
        try:
            if row is None or bool(ak.is_none(row, axis=0)):
                rows.append(None)
                continue
        except Exception:
            pass
        rows.append(row[int(choice[i])])
    return ak.unflatten(ak.Array(rows), 1)


def deep_index(v, idx):
    """Apply a hierarchy of scene/frame/molecule indexers to one field."""
    if not idx:
        return v
    first, rest = idx[0], idx[1:]
    v = v[first]
    batched = not isinstance(first, (int, np.integer, slice))
    for indexer in rest:
        if batched:
            v = _take_next_axis(v, indexer)
        else:
            v = v[indexer]
    return v


def _placeholder_like(reference: ak.Array, length: int) -> ak.Array:
    """Option-None array of ``length`` matching ``reference``'s inner type.

    Concatenating a missing field with ``ak.ones_like(toc)`` yields float
    placeholders. Mixing those with a string column such as
    ``molecule_enzyme_name`` produces an irreducible union that Awkward
    cannot jagged-index during molecule sampling.
    """
    if length <= 0:
        return ak.Array([])
    return ak.Array([None] * length)


def _apply_hierarchy_mask(array: ak.Array, mask: ak.Array) -> ak.Array:
    """Apply a nested boolean/integer mask, including optional columns.

    Optional fields such as ``residue_SASA`` are stored as an
    ``IndexedOptionArray`` whose content only contains scenes that have
    values. Indexing that content with a mask over every scene fails
    (inner ListArray length 402 vs slice length 1000). Compact both sides,
    apply the mask to present scenes, and scatter the result back into the
    original option slots.
    """
    layout = array.layout
    if not getattr(layout, "is_option", False):
        return array[mask]

    none = np.asarray(ak.to_numpy(ak.is_none(array, axis=0)))
    present = ak.to_packed(ak.drop_none(array))
    present_mask = ak.to_packed(mask[~none])
    updated = present[present_mask]
    index = np.full(len(array), -1, dtype=np.int64)
    index[~none] = np.arange(len(updated), dtype=np.int64)
    return ak.Array(
        ak.contents.IndexedOptionArray(
            ak.index.Index64(index),
            ak.to_layout(updated, allow_record=False),
        )
    )


class BatchProxy:
    """Attribute-access wrapper for nested :class:`Batch` hierarchy levels.

    ``BatchProxy`` tracks nesting depth so that attribute access like
    ``batch.molecules.residue_pos`` returns appropriately flattened Awkward
    arrays without manual indexing.
    """

    def __init__(self, batch: Batch, nesting: List[int] = []):
        self.batch = batch
        self.nesting = nesting
        self.mask_index = None

    def __getattr__(self, name: str) -> BatchProxy | ak.Array:
        if name in ["batch", "nesting", "mask_index"]:
            return super().__getattr__(name)
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
        if name in ["batch", "nesting", "mask_index"]:
            return super().__setattr__(name, value)
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
                    value = value.unflatten(  # type: ignore[attr-defined]
                        counts=self.batch.toc[self.batch.prefixes[axis]].ravel(),
                        axis=sum([n < axis for n in self.nesting]),
                    )
            self.batch.data[name] = value
        elif name.rstrip("s") in self.batch.prefixes and isinstance(value, BatchProxy):
            mask = value.mask_index
            prefix = name.rstrip("s")
            if mask is not None:
                if not isinstance(mask, ak.Array):
                    mask = ak.Array(mask)
                for axis in range(1, self.batch.prefixes.index(prefix) + 1)[::-1]:
                    if not axis in self.nesting:
                        mask = mask.unflatten(  # type: ignore[attr-defined]
                            counts=self.batch.toc[self.batch.prefixes[axis]].ravel(),
                            axis=sum([n < axis for n in self.nesting]),
                        )
                # mask every column of prefix and below
                for col in self.batch.data.keys():
                    if self.batch.prefixes.index(
                        col.split("_")[0]
                    ) >= self.batch.prefixes.index(prefix):
                        self.batch.data[col] = _apply_hierarchy_mask(
                            self.batch.data[col], mask
                        )
                # update toc
                if type(ak.ravel(mask)[0]) == np.bool_:  # todo: better type check
                    self.batch.toc[prefix] = ak.sum(mask, axis=-1)
                else:
                    self.batch.toc[prefix] = ak.num(mask, axis=-1)
        else:
            raise AttributeError(f"Attribute must start with a prefix.")

    def __getitem__(
        self, index: slice | int | ak.Array | Tuple[slice | int | ak.Array, ...]
    ) -> BatchProxy:
        self.mask_index = index
        return self


class Batch:
    """One shard of structured biomolecular data.

    A batch stores nested Awkward arrays keyed by ``{level}_{property}`` (e.g.
    ``residue_pos``, ``molecule_label``) together with a table-of-contents
    (``toc``) that records how many elements exist at each hierarchy level:
    scene → frame → molecule → chain → residue → atom.

    Batches support attribute-style access via :class:`BatchProxy` and fancy
    indexing to select subsets across the hierarchy.

    Examples
    --------
    .. code-block:: python

       batch = Batch({"residue_pos": pos, "residue_token": tokens})
       coords = batch.residues.residue_pos
       subset = batch[0]  # first scene
    """

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
                ref_col = _reference_column(data, prefix)
                axis = prefixes.index(prefix)
                counts = None
                if ref_col is not None:
                    try:
                        counts = ak.num(data[ref_col], axis=axis)
                    except Exception:
                        counts = None
                if counts is not None:
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
        if name in ["data", "prefixes", "levels", "toc", "length", "resolution"]:
            return super().__setattr__(name, value)
        if "_" in name and name.split("_")[0] in self.prefixes + ["vertex"]:
            return BatchProxy(self).__setattr__(name, value)
        elif name.rstrip("s") in self.prefixes:
            return BatchProxy(self).__setattr__(name, value)
        else:
            return super().__setattr__(name, value)

    def __getitem__(
        self, index: slice | int | ak.Array | Tuple[slice | int | ak.Array, ...]
    ) -> Batch:

        # the index is a tuple of indices for each axis
        if not isinstance(index, tuple):
            index = (index,)

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
            if field not in self_data:
                self_data[field] = _placeholder_like(other_data[field], len(self))
            if field not in other_data:
                other_data[field] = _placeholder_like(self_data[field], len(other))
            both_data[field] = ak.concatenate([self_data[field], other_data[field]])
        return Batch(both_data)

    def __radd__(self, other: Batch | Any) -> Batch:
        # if other is not a Batch, return a copy of self
        if not isinstance(other, Batch):
            data = {field: self.data[field] for field in self.data.keys()}
            return Batch(data)
        else:
            return other + self

    def filter_scenes(self, mask: ak.Array) -> Batch:
        """Return a batch containing only scenes where ``mask`` is True."""
        keep = ak.to_numpy(mask)
        data = {field: values[keep] for field, values in self.data.items()}
        return Batch(data, prefixes=self.prefixes, resolution=self.resolution)


class Split:
    """Train/validation/test partition assignments for dataset elements.

    Splits map named partitions (``"train"``, ``"val"``, ``"test"``, …) to
    integer labels per hierarchy level. Keys must end with ``_split`` and
    reference a valid level (``scene_split``, ``molecule_split``, etc.).

    Examples
    --------
    .. code-block:: python

       split = Split({"scene_split": ["train", "train", "val", "test"]})
       train_mask = split["train"]       # default split name
       val_mask = split["default", "val"]
    """

    def __init__(
        self,
        data: dict[str, ak.Array | np.ndarray | list | tuple] = {},
        default: str = "default",
    ):
        self.default = default
        self.data = {}
        self.names = {}
        for split_name, names in data.items():
            assert split_name.split("_")[-1] == "split" and split_name.split("_")[
                -2
            ] in [
                "scene",
                "frame",
                "molecule",
                "chain",
                "residue",
                "atom",
                "mutation",
            ], "Split name must end with 'x_split' where x is one of 'scene', 'frame', 'molecule', 'chain', 'residue', 'atom', or 'mutation'"
            names, inverse = np.unique(np.asarray(names), return_inverse=True)
            self.data[split_name] = inverse
            self.names[split_name] = list(names)

    def __getitem__(self, index) -> ak.Array:
        if isinstance(index, tuple):
            split_name, partition = index
        else:
            split_name = self.default
            partition = index
        if split_name == "default":
            split_name = self.default
        return self.data[split_name] == self.names[split_name].index(partition)

    def __add__(self, other: Split | dict) -> Split:
        if isinstance(other, dict):
            other = Split(other)
        return Split({**self.data, **other.data})

    def mask_filter(self, mask: np.ndarray) -> Split:
        """Return a split with entries selected by a boolean scene mask."""
        filtered = Split(default=self.default)
        filtered.data = {k: v[mask] for k, v in self.data.items()}
        filtered.names = {k: list(v) for k, v in self.names.items()}
        return filtered

    def save(self, path: Path) -> None:
        with open(path / "split.json", "w") as f:
            json.dump({k: v.tolist() for k, v in self.data.items()}, f)
        with open(path / "split.names.json", "w") as f:
            json.dump(self.names, f)
        with open(path / "split.default.json", "w") as f:
            json.dump(self.default, f)

    @classmethod
    def load(cls, path: Path) -> Split:
        path = Path(path)
        split = cls()
        json_path = path / "split.json"
        ak_path = path / "split.ak"
        if json_path.is_file():
            with open(json_path, "r") as f:
                split.data = {k: np.array(v) for k, v in json.load(f).items()}
            with open(path / "split.names.json", "r") as f:
                split.names = json.load(f)
            default_path = path / "split.default.json"
            if default_path.is_file():
                with open(default_path, "r") as f:
                    split.default = json.load(f)
            elif split.data:
                split.default = next(iter(split.data))
            return split
        if ak_path.is_file():
            with open(ak_path, "rb") as f:
                raw = pickle.load(f)
            with open(path / "split.names.json", "r") as f:
                names = json.load(f)
            values = np.asarray(ak.to_numpy(ak.ravel(ak.Array(raw))))
            if isinstance(names, list):
                key = "scene_split"
                split.names = {key: names}
                split.data = {key: values}
                split.default = key
            else:
                split.names = names
                if getattr(raw, "fields", None):
                    split.data = {
                        field: np.asarray(ak.to_numpy(ak.ravel(raw[field])))
                        for field in raw.fields
                    }
                else:
                    key = next(iter(names))
                    split.data = {key: values}
                split.default = next(iter(split.data))
            return split
        raise FileNotFoundError(f"No split.json or split.ak under {path}")


class Assets(dict):
    """Shared lookup tables referenced across dataset batches.

    Assets hold vocabularies, embedding matrices, ID maps, and other
    resources that are too large or too static to duplicate in every batch.
    Tasks and live transforms read from assets at load time.

    Examples
    --------
    .. code-block:: python

       assets = Assets({"residue_tokens": ["ALA", "GLY", ...], "residue_features": emb})
    """

    pass
