# Bioverse

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue?style=flat-square&logo=readthedocs&logoColor=white)](https://borgwardtlab.github.io/bioverse/)
[![PyPI](https://img.shields.io/pypi/v/bioverse-ml?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/bioverse-ml/)
[![Python](https://img.shields.io/pypi/pyversions/bioverse-ml?style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/bioverse-ml/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green?style=flat-square)](https://github.com/BorgwardtLab/bioverse/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-BorgwardtLab%2Fbioverse-black?style=flat-square&logo=github)](https://github.com/BorgwardtLab/bioverse)

**Bioverse** is a standardized framework for machine-learning experiments on
biomolecules — proteins, RNA, small molecules, and related structures. It
combines reusable benchmarks, transforms, and evaluation components with a
config-driven CLI so you can train and evaluate models without reimplementing
data loading, splitting, or metrics.

## Quick start

```bash
pip install bioverse-ml
pip install torch lightning   # required for training
bioverse train experiment.yaml
```

The default data directory is `~/.bioverse`. Override paths with environment
variables such as `BIOVERSE_ROOT` and `BIOVERSE_DATASET_ROOT`.

## Documentation

Full documentation is hosted on **GitHub Pages**:

**https://borgwardtlab.github.io/bioverse/**

- [Quickstart](https://borgwardtlab.github.io/bioverse/quickstart.html) — minimal
  `B_AFCATH` loader example
- [User Guide](https://borgwardtlab.github.io/bioverse/guides/user.html) —
  configuration and CLI workflows
- [Developer Guide](https://borgwardtlab.github.io/bioverse/guides/developer.html) —
  architecture and extension points
- [Implementations](https://borgwardtlab.github.io/bioverse/implementations/benchmarks.html) —
  auto-generated catalog of datasets, benchmarks, and components
- [Citation](https://borgwardtlab.github.io/bioverse/citation.html)

Build docs locally:

```bash
pip install -e ".[docs]"
cd docs && make html
python -m http.server --directory build/html 8080
```

## Pipeline

```text
Adapter  →  Dataset  →  Benchmark  →  Trainer + Model
                          (sampler,
                           task,
                           metric)
```

## Citation

If you use Bioverse in your research, please cite:

```bibtex
@software{bioverse2026,
  author = {Kucera, Tim and Bioverse Contributors},
  title = {Bioverse: A standardized framework for machine learning on biomolecules},
  year = {2026},
  url = {https://github.com/BorgwardtLab/bioverse}
}
```

See the [citation page](https://borgwardtlab.github.io/bioverse/citation.html) for details.

## License

BSD-3-Clause — see [LICENSE](LICENSE).
