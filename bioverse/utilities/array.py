from functools import partial
from typing import Any, Iterable

import awkward as ak
import numpy as np


class ArrayExtensions:
    def __getattr__(self, where: str) -> Any:
        if hasattr(type(self), where):
            return super().__getattribute__(where)
        elif hasattr(ak, where):
            return partial(getattr(ak, where), self)
        elif where.startswith("str_") and hasattr(ak.str, where[4:]):  # type: ignore
            return partial(getattr(ak.str, where[4:]), self)  # type: ignore
        else:
            if where in self._layout.fields:
                try:
                    return self[where]  # type: ignore
                except Exception as err:
                    raise AttributeError(
                        f"while trying to get field {where!r}, an exception "
                        f"occurred:\n{type(err)}: {err!s}"
                    ) from err
            else:
                raise AttributeError(f"no field named {where!r}")


ak.Array.__getattr__ = ArrayExtensions.__getattr__  # type: ignore


def flatten(
    data: ak.Array,
    axes: int | slice | Iterable[int] | None = None,
    exclude: int | slice | Iterable[int] | None = None,
) -> ak.Array:
    num_axes = data.ndim
    if axes is None:
        axes = list(range(num_axes))
    if exclude is None:
        exclude = []
    if isinstance(axes, int):
        axes = [axes]
    if isinstance(exclude, int):
        exclude = [exclude]
    if isinstance(axes, slice):
        axes = list(range(num_axes)[axes])
    if isinstance(exclude, slice):
        exclude = list(range(num_axes)[exclude])
    # replace negative axes with positive ones
    axes = [num_axes + axis if axis < 0 else axis for axis in axes]
    exclude = [num_axes + axis if axis < 0 else axis for axis in exclude]
    # flatten (start from last axis to not change axis indices during flattening)
    for axis in sorted(set(axes) - set(exclude))[::-1]:
        data = ak.flatten(data, axis=axis)
    return data


def onehot(data: ak.Array, num_classes: int) -> ak.Array:
    eye = ak.values_astype(np.eye(num_classes), int)

    def t(layout, **kwargs):
        if layout.is_numpy:
            return ak.contents.NumpyArray(eye[ak.values_astype(layout.data, int)])  # type: ignore

    return ak.transform(t, data)


def cumsum(array: ak.Array) -> ak.Array:
    sizes = ak.num(array, axis=-1)
    scan = ak.unflatten(np.cumsum(ak.ravel(array)), sizes, axis=0)
    return ak.fill_none(scan - ak.firsts(scan) + ak.firsts(array), [])


def diff(array: ak.Array) -> ak.Array:
    return ak.fill_none(array[..., 1:] - array[..., :-1], [])  # type: ignore
