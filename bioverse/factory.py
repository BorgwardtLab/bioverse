from importlib import import_module
from pathlib import Path
from typing import cast

from .benchmark import Benchmark
from .utilities import config, load


def BenchmarkFactory(cfg: str | Path | dict) -> Benchmark:

    # Load the configuration file
    if isinstance(cfg, str) or isinstance(cfg, Path):
        cfg = load(cfg)
        cfg = cast(dict, cfg)

    # Import the dataset, sampler, task, and metric and create instances
    for key in ["dataset", "sampler", "task", "metric"]:
        if isinstance(cfg[key], str):
            cfg[key] = getattr(import_module(f".{key}s", "bioverse"), cfg[key])()
        elif isinstance(cfg[key], dict):
            cfg[key] = getattr(
                import_module(f".{key}s", "bioverse"), cfg[key].pop("name")
            )(**cfg[key])
        else:
            raise ValueError("Invalid configuration for BenchmarkFactory.")

    # Import and create the transforms if present in the config
    if "transforms" in cfg:
        for i, transform in enumerate(cfg["transforms"]):
            if isinstance(transform, str):
                cfg["transforms"][i] = getattr(
                    import_module(".transforms", "bioverse"), transform
                )()
            elif isinstance(transform, dict):
                cfg["transforms"][i] = getattr(
                    import_module(".transforms", "bioverse"), transform.pop("name")
                )(**transform)
            else:
                raise ValueError("Invalid configuration for BenchmarkFactory.")

    # Create the benchmark class
    class BenchmarkInstance(Benchmark):
        name = cfg["name"]
        dataset = cfg["dataset"]
        sampler = cfg["sampler"]
        task = cfg["task"]
        metric = cfg["metric"]

    # Initialize the benchmark instance
    inst = BenchmarkInstance(root=config.custom_benchmarks_path)

    # Apply the transforms if present in the config
    if "transforms" in cfg:
        inst.apply(*cfg["transforms"])

    return inst
