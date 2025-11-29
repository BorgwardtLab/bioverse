import argparse
import importlib
import os
import sys

from omegaconf import OmegaConf

from .factory import BenchmarkFactory, TransformFactory
from .trainer import Trainer
from .utilities import config as CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("config", nargs="+")
    args = parser.parse_args()

    # load all config files
    config = [OmegaConf.load(c) for c in args.config if not "=" in c]
    # parse config options which were passed as arguments
    config += [OmegaConf.from_dotlist([c for c in args.config if "=" in c])]
    # merge into single config
    config = OmegaConf.merge(*config)
    # resolve all interpolations
    config = OmegaConf.to_container(config, resolve=True)

    assert "trainer" in config, "trainer must be specified in config"
    assert "model" in config, "model must be specified in config"
    assert "benchmark" in config, "benchmark must be specified in config"
    assert "transforms" in config, "transforms must be specified in config"

    # set global config options
    for key, value in config.get("globals", {}).items():
        setattr(CONFIG, key, value)

    # instantiate model
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    if isinstance(config["model"], dict):
        path, kwargs = next(iter(config["model"].items()))
    else:
        path, kwargs = config["model"], {}
    module, _, name = path.lstrip(".").rpartition(".")
    model = getattr(importlib.import_module(module), name)(**kwargs)

    # instantiate benchmark
    benchmark = BenchmarkFactory(config["benchmark"])
    transforms = TransformFactory(config["transforms"])
    benchmark.apply(transforms)

    # instantiate trainer
    trainer = Trainer(model, benchmark, **config["trainer"])

    # run trainer
    trainer.run(args.command)


if __name__ == "__main__":
    main()
