import importlib.resources
import os
import sys
from importlib import import_module
from pathlib import Path
from typing import cast

from .benchmark import Benchmark
from .dataset import ComposedDataset, Dataset
from .metric import MultiMetric
from .transform import Compose, Transform
from .utilities import load


def _load_config(cfg: str | Path | dict, submodule: str = None) -> dict:
    # load config resource provided with bioverse
    if isinstance(cfg, str) and not cfg.endswith(".yaml") and not submodule is None:
        cfg = Path(importlib.resources.files(f"bioverse.{submodule}") / (cfg + ".yaml"))
    # load config from user-provided file
    if isinstance(cfg, Path) or (isinstance(cfg, str) and cfg.endswith(".yaml")):
        name = Path(cfg).stem
        cfg = load(cfg)
        cfg["name"] = name
    return cfg


def _load_module(cfg: str | Path | dict, submodule: str = None) -> object:
    # load from user provided import path
    if isinstance(cfg, str) and cfg.startswith("."):
        cwd = os.getcwd()
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        module, _, name = cfg.lstrip(".").rpartition(".")
        return getattr(importlib.import_module(module), name)()
    # load from bioverse, provided as identifier without kwargs
    elif isinstance(cfg, str):
        return getattr(import_module(f"bioverse.{submodule}"), cfg)()
    # load from bioverse, provided as dict with kwargs
    elif isinstance(cfg, dict):
        name, kwargs = next(iter(cfg.items()))
        if name.startswith("."):
            cwd = os.getcwd()
            if cwd not in sys.path:
                sys.path.insert(0, cwd)
            module, _, name = name.lstrip(".").rpartition(".")
        else:
            module = f"bioverse.{submodule}"
        return getattr(import_module(module), name)(**(kwargs or {}))
    # load recursively from list
    elif isinstance(cfg, list):
        module_list = [_load_module(item, submodule) for item in cfg]
        if submodule == "metrics":
            return MultiMetric(module_list)
        else:
            return module_list
    else:
        raise ValueError("Invalid configuration.")


def BenchmarkFactory(cfg: str | Path | dict) -> Benchmark:
    # Load the configuration file
    cfg = _load_config(cfg, "benchmarks")

    # Create the benchmark class
    class BenchmarkInstance(Benchmark):
        name = cfg["name"]
        dataset = DatasetFactory(cfg["dataset"])
        sampler = _load_module(cfg["sampler"], "samplers")
        task = _load_module(cfg["task"], "tasks")
        metric = _load_module(cfg["metric"], "metrics")

    # Initialize the benchmark instance
    return BenchmarkInstance()


def TransformFactory(cfg: str | Path | list) -> Transform:
    if not isinstance(cfg, list):
        cfg = [cfg]
    transforms = []
    for transform in cfg:
        transform = _load_config(transform, None)
        transforms.append(_load_module(transform, "transforms"))
    return Compose(*transforms)


def DatasetFactory(cfg: str | Path | dict | list) -> Dataset:
    if isinstance(cfg, list):
        return ComposedDataset(DatasetFactory(item) for item in cfg)

    # Load the configuration file
    cfg = _load_config(cfg, "datasets")

    # Load transforms
    transforms = TransformFactory(cfg["transforms"] if "transforms" in cfg else [])

    # load parent
    if "parent" in cfg and "adapter" in cfg["parent"]:
        raise ValueError("Dataset cannot have both parent and adapter.")
    if "parent" in cfg:

        class DatasetInstance(Dataset):
            name = cfg["name"]

            def release(self):
                parent = DatasetFactory(cfg["parent"])
                return transforms(parent.shards, parent.split, parent.assets)

    elif "adapter" in cfg:
        if isinstance(cfg["adapter"], dict):
            name, kwargs = next(iter(cfg["adapter"].items()))
        else:
            name, kwargs = cfg["adapter"], {}
        adapter = _load_module(name, "adapters")

        class DatasetInstance(Dataset):
            name = cfg["name"]

            def release(self):
                batches, split, assets = adapter.download(**(kwargs or {}))
                return transforms(batches, split, assets)

    else:
        raise ValueError("Dataset must have either parent or adapter.")

    # Initialize the benchmark instance
    return DatasetInstance()
