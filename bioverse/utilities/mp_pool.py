"""Process pool that returns large payloads via /dev/shm instead of pipes.

``multiprocessing.Pool`` pickles every result through a single feeder thread.
Live-transform batches are tens of megabytes, so workers block on IPC, early
``join()`` hangs, and resource_tracker leaks semaphores. This pool sends only
tiny headers over queues and stores numeric payloads as numpy pickles in
shared memory.
"""

from __future__ import annotations

import os
import pickle
import tempfile
import time
import traceback
from itertools import count
from pathlib import Path
from queue import Empty
from typing import Any, Callable, Iterable, Iterator

import awkward as ak
import numpy as np

import multiprocess as mp

_WORKER_FN = None
_PACK = None
_COUNTER = count()
_POOL_PREFIX = None
_INLINE_LIMIT = 256 * 1024
_JOIN_TIMEOUT = 2.0
_GET_TIMEOUT = 1.0


def _mp_context():
    methods = mp.get_all_start_methods()
    forced = os.environ.get("BIOVERSE_MP_START")
    if forced in methods:
        return mp.get_context(forced)
    if "forkserver" in methods:
        return mp.get_context("forkserver")
    return mp.get_context("spawn")


def _shm_dir() -> Path:
    for candidate in ("/dev/shm", "/tmp"):
        path = Path(candidate)
        if path.exists() and os.access(path, os.W_OK):
            return path
    return Path(tempfile.gettempdir())


def _shm_path() -> str:
    prefix = _POOL_PREFIX or f"bioverse-{os.getpid()}"
    return str(_shm_dir() / f"{prefix}-{os.getpid()}-{next(_COUNTER)}.pkl")


_NUMERIC_KINDS = frozenset("biufc")


def _try_numpy(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        converted = value
    elif isinstance(value, ak.Array):
        try:
            converted = ak.to_numpy(value)
        except Exception:
            return None
    elif isinstance(value, (np.integer, np.floating, np.bool_)):
        converted = np.asarray(value)
    else:
        return None
    if converted.dtype.kind not in _NUMERIC_KINDS:
        return None
    return np.ascontiguousarray(converted)


def pack_blob(obj: Any) -> tuple:
    blob = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    if len(blob) <= _INLINE_LIMIT:
        return ("i", blob)
    path = _shm_path()
    with open(path, "wb") as handle:
        handle.write(blob)
    return ("f", path)


def unpack_blob(handle: tuple) -> Any:
    kind = handle[0]
    if kind == "i":
        return pickle.loads(handle[1])
    if kind == "f":
        path = handle[1]
        try:
            with open(path, "rb") as file:
                return pickle.loads(file.read())
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
    raise ValueError(f"Unknown payload kind {kind!r}")


def pack_data_object(data: Any) -> tuple:
    arrays: dict[str, np.ndarray] = {}
    other: dict[str, Any] = {}
    for key, value in vars(data).items():
        converted = _try_numpy(value)
        if converted is not None:
            arrays[key] = converted
        else:
            other[key] = value
    path = _shm_path()
    with open(path, "wb") as handle:
        pickle.dump((arrays, other), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return ("d", path)


def unpack_data_object(handle: tuple) -> Any:
    from bioverse.collaters.long import Data

    _, path = handle
    try:
        with open(path, "rb") as file:
            arrays, other = pickle.load(file)
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    return Data(**{**other, **arrays})


def pack_loader_result(result: Any) -> tuple:
    xy, data = result
    if data is None:
        return ("xy", pack_blob(xy))
    return ("data", pack_data_object(data))


def unpack_loader_result(handle: tuple) -> Any:
    kind = handle[0]
    if kind == "xy":
        return unpack_blob(handle[1]), None
    if kind == "data":
        data = unpack_data_object(handle[1])
        return (None, getattr(data, "y", None)), data
    raise ValueError(f"Unknown loader payload {kind!r}")


def _init_worker(fn: Callable, pack: Callable | None, prefix: str) -> None:
    global _WORKER_FN, _PACK, _POOL_PREFIX
    _WORKER_FN = fn
    _PACK = pack
    _POOL_PREFIX = prefix


def _worker_loop(in_q, out_q) -> None:
    while True:
        msg = in_q.get()
        if msg is None:
            break
        seq, item = msg
        try:
            result = _WORKER_FN(item)
            if _PACK is not None:
                result = _PACK(result)
            out_q.put((seq, True, result))
        except Exception as exc:
            out_q.put(
                (
                    seq,
                    False,
                    (type(exc).__name__, str(exc), traceback.format_exc()),
                )
            )


def _worker_main(fn: Callable, pack: Callable | None, prefix: str, in_q, out_q) -> None:
    _init_worker(fn, pack, prefix)
    _worker_loop(in_q, out_q)


def _drop_payload(payload: Any) -> None:
    try:
        if not isinstance(payload, tuple) or not payload:
            return
        kind = payload[0]
        if kind == "data":
            unpack_data_object(payload[1])
        elif kind == "xy":
            unpack_blob(payload[1])
        elif kind in {"i", "f"}:
            unpack_blob(payload)
        elif kind == "d":
            unpack_data_object(payload)
    except Exception:
        pass


def _kill_processes(processes, in_q, out_q) -> None:
    for queue in (in_q, out_q):
        try:
            queue.cancel_join_thread()
        except Exception:
            pass
    for _ in processes:
        try:
            in_q.put_nowait(None)
        except Exception:
            break
    deadline = time.time() + _JOIN_TIMEOUT
    for proc in processes:
        remaining = max(0.0, deadline - time.time())
        proc.join(timeout=remaining)
        if proc.is_alive():
            try:
                proc.kill()
            except Exception:
                pass
            proc.join(timeout=1.0)
    for queue in (in_q, out_q):
        try:
            queue.close()
        except Exception:
            pass


class WorkerPool(Iterator):
    """Ordered prefetching pool. Large results never travel over the pipe."""

    def __init__(
        self,
        fn: Callable,
        data: Iterable,
        workers: int,
        *,
        pack: Callable | None = pack_blob,
        unpack: Callable | None = unpack_blob,
        prefetch: int | None = None,
    ):
        self._unpack = unpack
        self._data = iter(data)
        self._prefetch = prefetch or max(workers * 4, 8)
        self._pending: dict[int, Any] = {}
        self._next_put = 0
        self._next_get = 0
        self._in_flight = 0
        self._exhausted = False
        self._closed = False
        self._prefix = f"bioverse-{os.getpid()}-{next(_COUNTER)}"
        ctx = _mp_context()
        self._in_q = ctx.Queue(maxsize=self._prefetch)
        self._out_q = ctx.Queue()
        self._processes = []
        for _ in range(workers):
            proc = ctx.Process(
                target=_worker_main,
                args=(fn, pack, self._prefix, self._in_q, self._out_q),
                daemon=True,
            )
            proc.start()
            self._processes.append(proc)

    def __iter__(self):
        return self

    def _alive(self) -> bool:
        return any(proc.is_alive() for proc in self._processes)

    def _submit(self) -> None:
        while not self._exhausted and self._in_flight < self._prefetch:
            try:
                item = next(self._data)
            except StopIteration:
                self._exhausted = True
                return
            self._in_q.put((self._next_put, item))
            self._next_put += 1
            self._in_flight += 1

    def _recv(self):
        while True:
            try:
                return self._out_q.get(timeout=_GET_TIMEOUT)
            except Empty:
                if self._in_flight <= 0:
                    raise RuntimeError("Loader workers produced no result.")
                if not self._alive():
                    raise RuntimeError("All loader workers exited unexpectedly.")

    def __next__(self):
        if self._closed:
            raise StopIteration
        try:
            self._submit()
            if self._in_flight == 0 and self._next_get not in self._pending:
                self.close()
                raise StopIteration
            while self._next_get not in self._pending:
                seq, ok, payload = self._recv()
                self._in_flight -= 1
                if not ok:
                    name, message, tb = payload
                    self.close()
                    raise RuntimeError(f"Worker failed with {name}: {message}\n{tb}")
                self._pending[seq] = payload
                self._submit()
            payload = self._pending.pop(self._next_get)
            self._next_get += 1
            self._submit()
            if self._unpack is not None:
                return self._unpack(payload)
            return payload
        except StopIteration:
            raise
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for payload in list(self._pending.values()):
            _drop_payload(payload)
        self._pending.clear()
        try:
            while True:
                _, ok, payload = self._out_q.get_nowait()
                if ok:
                    _drop_payload(payload)
        except Exception:
            pass
        _kill_processes(self._processes, self._in_q, self._out_q)
        self._processes = []
        try:
            while True:
                _, ok, payload = self._out_q.get_nowait()
                if ok:
                    _drop_payload(payload)
        except Exception:
            pass
        for path in _shm_dir().glob(f"{self._prefix}-*"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
