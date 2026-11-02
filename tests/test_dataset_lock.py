import os
import time
from multiprocessing import get_context
from pathlib import Path

from bioverse.data import Assets, Split
from bioverse.dataset import Dataset
from bioverse.transform import Transform
from bioverse.utilities import config
from bioverse.utilities.lock import exclusive_file_lock
from tests.dummy.dummy import dummy_batches

config.workers = 1


class _LockProbeDataset(Dataset):
    def release(self):
        return dummy_batches(4), Split(), Assets({})


class _OnceTransform(Transform):
    def __init__(self, log_path: str, delay: float = 0.3):
        self.log_path = log_path
        self.delay = delay

    def transform_batch(self, batch):
        time.sleep(self.delay)
        with open(self.log_path, "a") as handle:
            handle.write(f"{os.getpid()}\n")
        return batch


def _hold_lock(lock_path: str, started, finished, hold: float) -> None:
    with exclusive_file_lock(lock_path):
        started.set()
        time.sleep(hold)
        finished.set()


def _apply_worker(root: str, log_path: str) -> None:
    config.workers = 1
    dataset = _LockProbeDataset(root=root, version=1, online=False)
    dataset.apply(_OnceTransform(log_path, delay=0.4))


def test_exclusive_file_lock_blocks_other_process(tmp_path):
    ctx = get_context("fork")
    lock_path = tmp_path / "resource.lock"
    started = ctx.Event()
    finished = ctx.Event()
    holder = ctx.Process(
        target=_hold_lock, args=(str(lock_path), started, finished, 0.5)
    )
    holder.start()
    assert started.wait(timeout=5)
    t0 = time.monotonic()
    with exclusive_file_lock(lock_path):
        waited = time.monotonic() - t0
        assert finished.is_set()
        assert waited >= 0.4
    holder.join(timeout=5)
    assert holder.exitcode == 0


def test_apply_runs_once_with_concurrent_jobs(tmp_path):
    root = tmp_path / "datasets"
    log_path = tmp_path / "transform.log"
    _LockProbeDataset(root=root, version=1, online=False)

    ctx = get_context("fork")
    workers = [
        ctx.Process(target=_apply_worker, args=(str(root), str(log_path)))
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
        assert worker.exitcode == 0

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1

    dataset = _LockProbeDataset(root=root, version=1, online=False)
    dataset.apply(_OnceTransform(str(log_path), delay=0.4))
    assert log_path.read_text().strip().splitlines() == lines
    assert (Path(dataset.path) / "1.ak").is_file()
    assert (Path(dataset.path) / "num_shards.json").is_file()
