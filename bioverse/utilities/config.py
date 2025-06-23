import multiprocessing
import os
from pathlib import Path

seed = int.from_bytes("molecules are awesome".encode(), "little")
verbosity = 3
workers = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count())) * int(
    os.environ.get("SLURM_NTASKS_PER_CORE", 1)
)
cache_size = 5
root = Path(
    os.environ.get(
        "BIOVERSE_ROOT",
        Path.home() / ".bioverse",
    )
)
dataset_path = Path(
    os.environ.get(
        "BIOVERSE_DATASET_ROOT",
        root / "datasets",
    )
)
raw_path = Path(
    os.environ.get(
        "BIOVERSE_RAWDATA_ROOT",
        root / "raw",
    )
)
benchmarks_path = Path(
    os.environ.get(
        "BIOVERSE_BENCHMARKS_ROOT",
        root / "benchmarks",
    )
)
custom_benchmarks_path = Path(
    os.environ.get(
        "BIOVERSE_CUSTOM_BENCHMARKS_ROOT",
        root / "custom_benchmarks",
    )
)
thirdparty_path = Path(
    os.environ.get(
        "BIOVERSE_THIRDPARTY_ROOT",
        root / "thirdparty",
    )
)
scratch_path = Path(os.environ.get("BIOVERSE_SCRATCH_ROOT", "/tmp"))
if not scratch_path.exists():
    scratch_path.mkdir(parents=True, exist_ok=True)
