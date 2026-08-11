Developer Guide
===============

Overview
--------

This guide is for developers who want to **extend** Bioverse: add datasets,
benchmarks, transforms, tasks, or other components. If you only want to run
experiments with existing pieces, start with the :doc:`User Guide <user>`.

Contributing upstream is covered in the :doc:`Contributor Guide <contributor>`.

Architecture
------------

Bioverse follows a pipeline from raw data to evaluation:

.. code-block:: text

   Adapter.download()
        ↓
   Dataset (shards, split, assets)
        ↓  offline transforms
   Benchmark
        ├── Sampler.index()   → batch indices
        ├── Task.__call__()   → (features, targets)
        └── Metric.update()   → aggregated scores
        ↓
   Trainer + Model

**Adapters** fetch or generate raw data. **Datasets** persist transformed shards.
**Benchmarks** bind a dataset to a sampler, task, and metric. The **Trainer** drives
training and evaluation through a pluggable **backend** and **collater**.

Core abstractions
-----------------

.. list-table::
   :header-rows: 1

   * - Class
     - Role
   * - :class:`~bioverse.adapter.Adapter`
     - Download raw data; return batches, split, and assets
   * - :class:`~bioverse.dataset.Dataset`
     - Versioned on-disk dataset built from an adapter or parent dataset
   * - :class:`~bioverse.transform.Transform`
     - Modify batches, splits, and assets (offline or live)
   * - :class:`~bioverse.sampler.Sampler`
     - Map table-of-contents rows to batch indices
   * - :class:`~bioverse.task.Task`
     - Extract model inputs and targets from a virtual batch
   * - :class:`~bioverse.metric.Metric`
     - Accumulate predictions and compute scores
   * - :class:`~bioverse.benchmark.Benchmark`
     - Orchestrate dataset, sampler, task, and metric

API reference pages document each base class in detail under
:doc:`../api_reference/code_structure`.

Data model
----------

Bioverse represents structured biomolecular data with **Awkward Array** records.

- :class:`~bioverse.data.Batch` — one shard of scenes/frames/molecules/atoms
- :class:`~bioverse.data.Split` — train/val/test partition assignments
- :class:`~bioverse.data.Assets` — auxiliary lookup tables
- :class:`~bioverse.virtual.VirtualBatch` — lazy view over on-disk shards for the task

Resolution flows from scenes → frames → molecules → residues → atoms. Tasks and
samplers refer to this hierarchy when indexing and batching.

Configuration and factories
---------------------------

:mod:`bioverse.factory` resolves YAML configs and class names at runtime:

- :func:`~bioverse.factory.DatasetFactory` — builds a :class:`~bioverse.dataset.Dataset`
  from ``D_*.yaml`` (adapter or parent dataset + transforms)
- :func:`~bioverse.factory.BenchmarkFactory` — builds a :class:`~bioverse.benchmark.Benchmark`
  from ``B_*.yaml``
- :func:`~bioverse.factory.TransformFactory` — composes transforms from a list

Class names in YAML (``BinaryAccuracyMetric``, ``MoleculeSampler``, …) are instantiated via
lazy imports in each subpackage's ``__getattr__``. Dict entries pass constructor kwargs:

.. code-block:: yaml

   task:
     PropertyPredictionTask:
       property: mu

Paths starting with ``.`` load classes from the current working directory instead
of ``bioverse.<subpackage>``.

Package layout
--------------

.. code-block:: text

   bioverse/
   ├── adapters/       # Adapter implementations
   ├── processors/     # File format processors (PDB, CIF, …)
   ├── datasets/       # D_*.yaml dataset configs
   ├── transforms/     # Transform implementations
   ├── samplers/       # Sampler implementations
   ├── tasks/          # Task implementations
   ├── metrics/        # Metric implementations
   ├── benchmarks/     # B_*.yaml benchmark configs
   ├── backends/       # TorchBackend, TensorflowBackend
   ├── collaters/      # LongCollater, WideCollater, …
   ├── factory.py      # Config resolution
   ├── cli.py          # bioverse CLI
   └── trainer.py      # Training loop

Each implementation subpackage discovers modules from filenames and exposes classes
through :func:`__getattr__` (see ``bioverse/adapters/__init__.py`` for the pattern).

Built-in vs custom components
-----------------------------

.. list-table::
   :header-rows: 1

   * - Component
     - Typical form
   * - Dataset
     - ``D_*.yaml`` (adapter or parent + transforms)
   * - Benchmark
     - ``B_*.yaml`` (references dataset, sampler, task, metric)
   * - Transform, Sampler, Task, Metric, Adapter, Processor
     - Python subclass in the matching subpackage

Creating new components
-----------------------

Step-by-step guides live under **How to**:

- :doc:`../tutorials/make_your_own_dataset`
- :doc:`../tutorials/make_your_own_benchmark`
- :doc:`../tutorials/make_your_own_metric`
- :doc:`../tutorials/make_your_own_sampler`
- :doc:`../tutorials/make_your_own_processor`
- :doc:`../tutorials/make_your_own_task`
- :doc:`../tutorials/make_your_own_transform`

In general:

1. Subclass the appropriate base class
2. Place the module in the correct subpackage (or reference it with a ``.`` import path)
3. For datasets and benchmarks, add a YAML config
4. Test with :func:`~bioverse.factory.BenchmarkFactory` before wiring a full experiment
5. Document via docstrings — implementation pages are generated automatically

Integrating with the CLI
------------------------

The CLI (:mod:`bioverse.cli`) expects your **model** to live outside the package.
Implementation classes (transforms, metrics, …) must be importable as
``bioverse.<subpackage>.<ClassName>`` or via a ``.`` path for local prototypes.

Experiment YAML is merged with OmegaConf; any component referenced by name must be
instantiable without side effects in ``__init__``.

Testing locally
---------------

Validate a new benchmark before training:

.. code-block:: python

   from bioverse.factory import BenchmarkFactory

   benchmark = BenchmarkFactory("B_MYBENCH")
   benchmark.apply()  # if you added offline transforms via experiment config
   loader = benchmark.loader(partition="train", batch_size=2, progress=True)
   (X, y), data = next(iter(loader))
   print(X, y)

Add pytest tests under ``tests/`` following existing patterns (see
:doc:`contributor`).

Documentation
-------------

Python implementations under ``bioverse/<subpackage>/`` appear automatically on
the matching :doc:`../implementations/adapters` page once the class subclasses
the correct base type. YAML configs in ``bioverse/datasets/`` and
``bioverse/benchmarks/`` are listed on the datasets and benchmarks
implementation pages; add ``description`` and ``citation`` keys for rendered
summaries (ignored at runtime).

Rebuild docs with ``make html`` in ``docs/``.
