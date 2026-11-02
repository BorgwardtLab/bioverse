"""Inter-process locks for dataset materialization on shared filesystems."""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .io import note


@contextmanager
def exclusive_file_lock(
    lock_path: Path | str,
    wait_message: str | None = None,
) -> Iterator[None]:
    """Hold an exclusive POSIX lock until the context exits.

    ``fcntl.lockf`` is released when the process dies, including ``kill -9``,
    so a crashed writer cannot leave a stale lock. On GPFS the lock is visible
    cluster-wide, which is what concurrent Slurm jobs need.
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if wait_message:
                note(wait_message)
            fcntl.lockf(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.lockf(fd, fcntl.LOCK_UN)
        os.close(fd)
